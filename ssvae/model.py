import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from utils.custom_mlp import MLP, Exp


class SSVAE(nn.Module):
    """
    Semi-supervised VAE for HEA phase prediction.

    Generative model:
      p(z) = N(0, I)
      p(y) = Bernoulli(0.5)                    # phase label prior
      p(x | y, z) = Multinomial(decoder(y, z)) # alloy composition

    Inference model (guide):
      q(y | e) = Bernoulli(encoder_y(e))        # phase from engineered features
      q(z | x, y) = N(encoder_z(x, y))          # latent from composition + phase
    """

    def __init__(
        self,
        output_size=1,
        input_size=30,
        z_dim=2,
        hidden_layers=(100, 100),
        use_cuda=False,
        aux_loss_multiplier=None,
    ):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = list(hidden_layers)
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self._setup_networks()

    def _setup_networks(self):
        h = self.hidden_layers

        # Predicts single-phase probability from 8 engineered features
        self.encoder_y = MLP(
            [8] + h + [1],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            use_cuda=self.use_cuda,
        )

        # Encodes (composition, phase label) -> (z_mean, z_std)
        self.encoder_z = MLP(
            [self.input_size + 1] + h + [[self.z_dim, self.z_dim]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            use_cuda=self.use_cuda,
        )

        # Decodes (z, phase label) -> composition probabilities
        self.decoder = MLP(
            [self.z_dim + 1] + h + [self.input_size],
            activation=nn.Softplus,
            output_activation=nn.Softmax(dim=-1),
            use_cuda=self.use_cuda,
        )

        if self.use_cuda:
            self.cuda()

    def model(self, xs, es=None, ys=None):
        """
        Generative model p(x, y, z):
          z ~ N(0, I)
          y ~ Bernoulli(0.5)
          x ~ Multinomial(decoder(z, y))
        """
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)
        with pyro.plate("data"):
            prior_loc = torch.zeros(batch_size, self.z_dim, device=xs.device)
            prior_scale = torch.ones(batch_size, self.z_dim, device=xs.device)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            ys_prior = torch.ones(batch_size, self.output_size, device=xs.device) * 0.5
            if ys is None:
                ys = pyro.sample("y", dist.Bernoulli(probs=ys_prior).to_event(1))
            else:
                ys = pyro.sample("y", dist.Bernoulli(probs=ys_prior).to_event(1), obs=ys)

            loc = self.decoder([zs, ys])
            pyro.sample("x", dist.Multinomial(total_count=101, probs=loc), obs=xs)
            return loc

    def guide(self, xs, es=None, ys=None):
        """
        Variational posterior q(z, y | x, e):
          y ~ Bernoulli(encoder_y(e))       if label unknown
          z ~ N(encoder_z(x, y))
        """
        with pyro.plate("data"):
            if ys is None:
                probs = self.encoder_y(es)
                ys = pyro.sample("y", dist.Bernoulli(probs=probs).to_event(1))
            loc, scale = self.encoder_z([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def classifier(self, es):
        """Predict binary phase label (0=multi-phase, 1=single-phase) from engineered features."""
        return (self.encoder_y(es) > 0.5).float()

    def predict_proba(self, es):
        """Return single-phase probability as numpy array (for SHAP compatibility)."""
        if not torch.is_tensor(es):
            es = torch.tensor(es, dtype=torch.float32)
            if self.use_cuda:
                es = es.cuda()
        return self.encoder_y(es).detach().cpu().numpy()

    def model_classify(self, xs, es, ys=None):
        """Auxiliary supervised classification loss (Kingma et al. 2014)."""
        pyro.module("ss_vae", self)
        with pyro.plate("data"):
            if ys is not None:
                probs = self.encoder_y(es)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.Bernoulli(probs=probs).to_event(1), obs=ys)

    def guide_classify(self, xs, es, ys=None):
        pass
