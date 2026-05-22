import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. Removed device args from init
2. Changed embed_dim to llm_dim
3. Add vision_dim argument
4. add MLP projection vit_dim --> llm_dim
5: Make selector return attention mask in train (how it "removes" tokens)
    # NOTE: this behavior is not expected in forward. Make an if statement dependent on arch type in forward.
# 6: changed parts of prismatic.py
"""
    

class DenseGumbelAttentionSelector(nn.Module):
    """
    This module estimates the importance of each token and divides the
    sequence dimension into two sets.

    1. Computes context-aware representations via MultiheadAttention.
    2. Projects to importance logits (Keep vs. Compress).
    3. Uses Gumbel-Softmax for differentiable binary masking.
    """
    def __init__(self, llm_dim: int, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        # New: Maps contextualized tokens to two logits: [Keep, Compress]
        self.router = nn.Linear(llm_dim, 2)

    def forward(self, x, tau=1.0):
        B, N, D = x.shape

        # 1. Compute attention-enriched token representations
        # We don't need the raw attention weights anymore, just the output features
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=False
        ) 

        # Residual connection keeps original token identity intact for the router
        context_x = x + attn_out

        # 2. Compute Routing Logits
        logits = self.router(context_x) # Shape: (B, N, 2)

        # 3. Differentiable Binary Selection (Straight-Through Estimator)
        # hard=True forces the output to be exactly 0 or 1 for the forward pass,
        # but uses the continuous gradients for the backward pass.
        gumbel_out = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1) # (B, N, 2)

        # 4. Extract continuous-differentiable masks
        keep_mask = gumbel_out[:, :, 0:1]     # (B, N, 1)
        compress_mask = gumbel_out[:, :, 1:2] # (B, N, 1)

        # 5. Mask the sets (Zeroing out instead of gathering)
        # Sequence length N is maintained for training
        k_set = x * keep_mask
        q_set = x * compress_mask

        # 6. Extract soft probabilities for L1 Regularization in your training loop
        # We use standard softmax here because we want the true continuous probability
        keep_probs = F.softmax(logits, dim=-1)[:, :, 0] # (B, N)

        # We return the sets, the masks (for downstream attention blocking), and the probs
        return k_set, q_set, keep_mask, compress_mask, keep_probs

class SelectorCompressorPipeline(nn.Module):
    """
    1. Routes tokens using DenseGumbelAttentionSelector.
    2. Bypasses selected tokens.
    3. Compresses rejected tokens using Learnable Queries (Q-Former).
    4. Feeds the combined sequence into a downstream Transformer.
    """
    def __init__(self, vision_dim, llm_dim, num_heads, num_compressed_tokens):
        super().__init__()
        # 0. Vision --> LLM projector
        self.vit2llm = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        # 1. The Selector
        self.selector = DenseGumbelAttentionSelector(llm_dim, num_heads)
        
        # 2. The Compressor (Learnable Queries + Cross Attention)
        self.num_compressed_tokens = num_compressed_tokens
        self.compress_queries = nn.Parameter(torch.randn(1, num_compressed_tokens, llm_dim))
        self.compress_mha = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Downstream Connector (Standard Transformer Layer)
        self.connector = nn.TransformerEncoderLayer(d_model=llm_dim, nhead=num_heads, batch_first=True)
        
        self.tau = 1.0
        self.inference_k = 128
        self.latest_keep_probs = None

        # === Ablation routing (eval-time only) ===
        # route_mode in {"selector", "full", "random_topk", "oracle"}
        # use_qformer toggles the compress_mha path
        # _oracle_indices: optional LongTensor (B, k) set by the eval harness per batch
        # latest_selected_indices: LongTensor (B, k) saved on the "selector" path for IoU
        self.route_mode = "selector"
        self.use_qformer = True
        self._oracle_indices = None
        self.latest_selected_indices = None

    def forward(self, x):
        """
        Automatically routes to the correct logic based on model.train() or model.eval()
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_inference(
                x,
                k=self.inference_k,
                route_mode=self.route_mode,
                use_qformer=self.use_qformer,
                oracle_indices=self._oracle_indices,
            )

    @staticmethod
    def _gather_split(x, sorted_idx, k):
        """Split x along seq dim into (kept = first k rows of sorted_idx, dropped = rest)."""
        B, N, D = x.shape
        kept_raw = sorted_idx[:, :k]
        dropped_raw = sorted_idx[:, k:]
        kept = torch.gather(x, dim=1, index=kept_raw.unsqueeze(-1).expand(-1, -1, D))
        dropped = torch.gather(x, dim=1, index=dropped_raw.unsqueeze(-1).expand(-1, -1, D))
        return kept, dropped, kept_raw

    def _forward_train(self, x):
        # 0: Project vit dim to llm dim
        x = self.vit2llm(x)

        B, N, D = x.shape
        
        # 1. Get routing masks (Shape: B, N, 1) and probs
        _, _, keep_mask, compress_mask, keep_probs = self.selector(x, self.tau)

        # 2. BYPASS PATH (Soft Masking)
        # Keep tensor size (B, N, D), zero out rejected tokens. 
        # Gradients flow through this multiplication.
        bypassed_tokens = x * keep_mask 

        # 3. COMPRESSION PATH (Soft Masking)
        queries = self.compress_queries.expand(B, -1, -1) # (B, M, D)
        tokens_to_compress = x * compress_mask            # (B, N, D)
        
        # We must tell PyTorch MHA to ignore the zeroed-out tokens.
        # key_padding_mask expects True for tokens to IGNORE.
        compress_padding_mask = (compress_mask.squeeze(-1) == 0) # (B, N)

        compressed_tokens, _ = self.compress_mha(
            query=queries, 
            key=tokens_to_compress, 
            value=tokens_to_compress, 
            key_padding_mask=compress_padding_mask
        ) # Output Shape: (B, M, D)

        # 4. RECOMBINE FOR CONNECTOR
        # Combine the padded sequence and the M compressed tokens
        combined_sequence = torch.cat([bypassed_tokens, compressed_tokens], dim=1) # (B, N + M, D)

        # Create a padding mask for the downstream connector
        # keep_mask is 1 (keep). Compressed tokens are always kept (1).
        compressed_mask_ones = torch.ones(B, self.num_compressed_tokens, device=x.device)
        connector_mask = torch.cat([keep_mask.squeeze(-1), compressed_mask_ones], dim=1) # (B, N + M)
        
        connector_padding_mask = (connector_mask == 0) # True for tokens to ignore

        # 5. Execute downstream connector
        final_output = self.connector(combined_sequence, src_key_padding_mask=connector_padding_mask)
        self.latest_keep_probs = keep_probs
        return final_output, connector_mask.bool()


    @torch.no_grad()
    def _forward_inference(
        self,
        x,
        k=128,
        route_mode="selector",
        use_qformer=True,
        oracle_indices=None,
    ):
        """
        Inference path with ablation routing. Always runs `vit2llm` then `connector`;
        the middle (which tokens are kept vs. compressed) depends on `route_mode`
        and `use_qformer`.

        route_mode:
          - "selector":    Trained Gumbel selector picks top-k.
          - "full":        Keep all N tokens (no selection).
          - "random_topk": Randomly pick k tokens.
          - "oracle":      Use externally provided `oracle_indices` (B, k).

        use_qformer:
          If True, dropped tokens go through `compress_mha` against learned queries
          and the M compressed tokens are concatenated to the kept tokens.
          If False, only kept tokens are forwarded.
        """
        x = self.vit2llm(x)
        B, N, D = x.shape

        self.latest_keep_probs = None
        self.latest_selected_indices = None

        # ---- Determine kept / dropped token splits ----
        if route_mode == "full":
            kept = x
            dropped = x.new_zeros(B, 0, D)

        elif route_mode == "random_topk":
            actual_k = min(k, N)
            perm = torch.stack(
                [torch.randperm(N, device=x.device) for _ in range(B)], dim=0
            )  # (B, N)
            kept, dropped, _ = self._gather_split(x, perm, actual_k)

        elif route_mode == "oracle":
            assert oracle_indices is not None, "`oracle` route_mode requires `oracle_indices`"
            assert oracle_indices.dim() == 2 and oracle_indices.size(0) == B, (
                f"oracle_indices must be (B, k); got {tuple(oracle_indices.shape)}"
            )
            oi = oracle_indices.to(x.device, dtype=torch.long)
            actual_k = oi.size(1)
            # Build the complement so we can optionally feed it to the Q-Former.
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            mask.scatter_(1, oi, False)
            complement = torch.stack([mask[b].nonzero(as_tuple=False).squeeze(-1) for b in range(B)], dim=0)
            sorted_idx = torch.cat([oi, complement], dim=1)
            kept, dropped, _ = self._gather_split(x, sorted_idx, actual_k)

        elif route_mode == "selector":
            actual_k = min(k, N)
            logits = self.selector.router(x)
            keep_probs = F.softmax(logits, dim=-1)[:, :, 0]
            _, sorted_idx = torch.sort(keep_probs, dim=1, descending=True)
            kept, dropped, kept_raw = self._gather_split(x, sorted_idx, actual_k)
            self.latest_selected_indices = kept_raw

        else:
            raise ValueError(f"Unknown route_mode `{route_mode}`")

        # ---- Optional Q-Former compression of dropped tokens ----
        if use_qformer:
            queries = self.compress_queries.expand(B, -1, -1)
            if dropped.size(1) > 0:
                compressed, _ = self.compress_mha(
                    query=queries, key=dropped, value=dropped
                )
            else:
                compressed = torch.zeros(
                    B, self.num_compressed_tokens, D, device=x.device, dtype=x.dtype
                )
            combined = torch.cat([kept, compressed], dim=1)
        else:
            combined = kept

        return self.connector(combined)