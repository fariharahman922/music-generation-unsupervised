import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive music token generation.

    Input:
        token_ids: (batch_size, seq_len)
        genre_ids: optional (batch_size,)

    Output:
        logits: (batch_size, seq_len, vocab_size)

    This follows the Task 3 idea in the PDF:
        p(X) = product_t p(x_t | x_<t)

    and optionally supports:
        h_t = Emb(x_t) + Emb(genre)
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len=256,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        num_genres=None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_genres = num_genres

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional genre embedding to align with roadmap table
        if num_genres is not None:
            self.genre_embedding = nn.Embedding(num_genres, d_model)
        else:
            self.genre_embedding = None

        self.dropout = nn.Dropout(dropout)

        # Decoder-only transformer using encoder blocks + causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _build_causal_mask(self, seq_len, device):
        """
        Causal mask so position t can attend only to <= t.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def forward(self, token_ids, genre_ids=None):
        """
        Forward pass.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
            genre_ids: optional Tensor of shape (batch_size,)

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        if self.genre_embedding is not None and genre_ids is not None:
            genre_emb = self.genre_embedding(genre_ids).unsqueeze(1)  # (batch, 1, d_model)
            x = x + genre_emb

        x = self.dropout(x)

        causal_mask = self._build_causal_mask(seq_len, device=device)

        x = self.transformer(x, mask=causal_mask)
        x = self.final_norm(x)

        logits = self.output_layer(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        max_new_tokens=128,
        genre_ids=None,
        temperature=1.0,
        top_k=None,
        eos_token_id=None
    ):
        """
        Autoregressive generation:
            x_t ~ p_theta(x_t | x_<t)

        Args:
            start_tokens: Tensor of shape (batch_size, start_len)
            max_new_tokens: number of new tokens to generate
            genre_ids: optional Tensor of shape (batch_size,)
            temperature: sampling temperature
            top_k: if set, sample only from top-k logits
            eos_token_id: stop early if all sequences hit EOS

        Returns:
            generated: Tensor of shape (batch_size, start_len + generated_len)
        """
        self.eval()

        generated = start_tokens

        for _ in range(max_new_tokens):
            if generated.size(1) > self.max_seq_len:
                generated_input = generated[:, -self.max_seq_len:]
            else:
                generated_input = generated

            logits = self.forward(generated_input, genre_ids=genre_ids)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            if temperature <= 0:
                raise ValueError("temperature must be > 0")

            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                values, indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                next_token_logits = filtered_logits

            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            generated = torch.cat([generated, next_tokens], dim=1)

            if eos_token_id is not None:
             
                if torch.all(next_tokens.squeeze(1) == eos_token_id):
                    break

        return generated


if __name__ == "__main__":
    batch_size = 4
    seq_len = 256
    vocab_size = 195

    model = MusicTransformer(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_genres=None
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(token_ids)

    print("Input token shape:", token_ids.shape)
    print("Logits shape:", logits.shape)

    start_tokens = torch.randint(0, vocab_size, (batch_size, 16))
    generated = model.generate(
        start_tokens=start_tokens,
        max_new_tokens=32,
        temperature=1.0,
        top_k=20
    )
    print("Generated token shape:", generated.shape)