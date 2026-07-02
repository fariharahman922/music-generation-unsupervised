import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for piano-roll sequences.

    Expected input shape:
        x: (batch_size, seq_len, input_dim)

    For your project:
        seq_len = 128
        input_dim = 88
    """

    def __init__(
        self,
        input_dim=88,
        hidden_dim=256,
        latent_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: f_phi(X) -> z
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: g_theta(z) -> X_hat
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Sigmoid()

    def encode(self, x):
        """
        Encode input sequence into latent vector z.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            z: Tensor of shape (batch_size, latent_dim)
        """
        _, (h_n, _) = self.encoder_lstm(x)

        last_hidden = h_n[-1]
        z = self.hidden_to_latent(last_hidden)

        return z

    def decode(self, z, seq_len):
        """
        Decode latent vector z into reconstructed sequence.

        Args:
            z: Tensor of shape (batch_size, latent_dim)
            seq_len: int

        Returns:
            x_hat: Tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)

        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)

        h0 = self.latent_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.latent_to_cell(z).unsqueeze(0).repeat(self.num_layers, 1, 1)

        decoder_output, _ = self.decoder_lstm(decoder_input, (h0, c0))
        x_hat = self.output_layer(decoder_output)
        x_hat = self.output_activation(x_hat)

        return x_hat

    def forward(self, x):
        """
        Full autoencoder forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            x_hat: reconstructed sequence, shape (batch_size, seq_len, input_dim)
            z: latent vector, shape (batch_size, latent_dim)
        """
        seq_len = x.size(1)
        z = self.encode(x)
        x_hat = self.decode(z, seq_len)

        return x_hat, z


if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    input_dim = 88

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=128,
        num_layers=2,
        dropout=0.2
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    x_hat, z = model(x)

    print("Input shape:", x.shape)
    print("Reconstruction shape:", x_hat.shape)
    print("Latent shape:", z.shape)