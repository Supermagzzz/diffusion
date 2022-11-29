from config import *

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        self.seq_len = seq_len

        self.pos_embed_learned_table = nn.Embedding(self.seq_len, HIDDEN)
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU()
        )
        self.unite_pos = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

    def forward(self, batch_size):
        pos_embed = torch.Tensor([i for i in range(self.seq_len)]).long().to(device)
        pos_embed = torch.cat([self.pos_embed_table(pos_embed), self.pos_embed_learned_table(pos_embed)], dim=-1)
        pos_embed = self.unite_pos(pos_embed)
        return make_batch(batch_size, pos_embed)


class SvgEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = PositionalEncoder(N * M_REAL // 6)
        self.float_embed = nn.Embedding(BLOCKS, HIDDEN)

        self.segment_embed = nn.Sequential(
            nn.Linear(HIDDEN * 6, HIDDEN),
            nn.ReLU()
        )

        self.make_input_embed = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

    def forward(self, svg):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, N * M_REAL)
        svg_long = torch.clamp((svg + SEQ_RANGE) / (2 * SEQ_RANGE) * BLOCKS, 0,
                               BLOCKS - 1).long()

        encoded_coords = self.float_embed(svg_long)
        segment_coords = encoded_coords.reshape(batch_size, N * M_REAL // 6, HIDDEN * 6)
        segment_embeds = self.segment_embed(segment_coords)

        pos_embed = self.pos_embed(batch_size)

        embeds = self.make_input_embed(torch.cat([segment_embeds, pos_embed], dim=-1))

        return embeds


class VariationalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.svg_encoder = SvgEncoder()

        self.mean_token = nn.Embedding(1, HIDDEN)
        self.var_token = nn.Embedding(1, HIDDEN)
        self.encoder = BertModel(BertConfig(
            BLOCKS,
            HIDDEN,
            num_attention_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=N * M_REAL // 6 + 2
        ))

        self.make_mean = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN)
        )
        self.make_var = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN)
        )

        self.N = torch.distributions.Normal(0, 1)
        if device == torch.device("cuda"):
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, svg):
        batch_size = svg.shape[0]
        embeds = self.svg_encoder(svg)
        mean_token = make_batch(batch_size, self.mean_token(torch.LongTensor([0]).to(device)))
        var_token = make_batch(batch_size, self.var_token(torch.LongTensor([0]).to(device)))
        hidden_state = self.encoder(inputs_embeds=torch.cat([mean_token, var_token, embeds], dim=1)).last_hidden_state
        z_mean = self.make_mean(hidden_state[:, 0, :])
        z_var = torch.exp(self.make_var(hidden_state[:, 1, :]))
        z = z_mean + z_var * self.N.sample(z_mean.shape).to(device)
        kl = (z_var ** 2 + z_mean ** 2 - torch.log(z_var) - 1 / 2).mean()
        return z, kl


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = PositionalEncoder(N * M_REAL // 6)

        self.make_input_embeds = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

        self.decoder = BertModel(BertConfig(
            BLOCKS,
            HIDDEN,
            num_attention_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=N * M_REAL // 6
        ))

        self.make_segment_embed = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN * 6),
            nn.ReLU(),
            nn.Linear(HIDDEN * 6, HIDDEN * 6),
        )

        self.make_float_embed = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        seq_len = N * M_REAL // 6

        pos_embeds = self.pos_embed(batch_size)
        hidden_embeds = make_seq(seq_len, hidden)

        embeds = self.make_input_embeds(torch.cat([
            pos_embeds,
            hidden_embeds
        ], dim=-1))

        embeds = self.decoder(inputs_embeds=embeds).last_hidden_state

        segment_embed = self.make_segment_embed(embeds)
        segment_embed = segment_embed.reshape(batch_size, seq_len * 6, HIDDEN)

        float_embed = self.make_float_embed(segment_embed)

        return float_embed.reshape(batch_size, N, M_REAL)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.svg_encoder = SvgEncoder()

        self.cls_token = nn.Embedding(1, HIDDEN)
        self.encoder = BertModel(BertConfig(
            BLOCKS,
            HIDDEN,
            num_attention_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=N * M_REAL // 6 + 1
        ))

        self.make_answer = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, svg):
        batch_size = svg.shape[0]
        embeds = self.svg_encoder(svg)
        cls_token = make_batch(batch_size, self.cls_token(torch.LongTensor([0]).to(device)))
        hidden_state = self.encoder(inputs_embeds=torch.cat([cls_token, embeds], dim=1)).last_hidden_state
        answer = self.make_answer(hidden_state[:, 0, :])
        answer = answer.reshape(batch_size)
        return torch.sigmoid(answer)


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.to(device)
        z, kl = self.encoder(x)
        return self.decoder(z), kl

