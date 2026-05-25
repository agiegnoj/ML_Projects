import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from tqdm import tqdm

class Transformer(nn.Module):
    def __init__(self, vocab, modelDim=512, numHeads=8, numLayers=6, maxSeqLength=120):
        super().__init__()
        self.modelDim = modelDim
        
        vocabSize = len(vocab)

        self.embedding = nn.Embedding(vocabSize, modelDim)
        self.positionalEmbedding = nn.Embedding(maxSeqLength, modelDim)

        self.encoder = nn.ModuleList([EncoderLayer(modelDim, numHeads) for _ in range(numLayers)])
        self.decoder = nn.ModuleList([DecoderLayer(modelDim, numHeads) for _ in range(numLayers)])

        self.outputLayer = nn.Linear(modelDim, vocabSize)

    def forward(self, srcTokens, tgtTokens, padTokenId):
        srcEmb = self.addPositionalEmbedding(self.embedding(srcTokens))
        tgtEmb = self.addPositionalEmbedding(self.embedding(tgtTokens))

        srcMask = self.paddingMask(srcTokens, padTokenId)
        tgtMask = self.decoderMask(tgtTokens, padTokenId)

        encOut = srcEmb
        for layer in self.encoder:
            encOut = layer(encOut, srcMask)

        decOut = tgtEmb
        for layer in self.decoder:
            decOut = layer(decOut, encOut, tgtMask, srcMask)

        return self.outputLayer(decOut)

    def fit(self, dl, padId, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=padId)
        description= "started training"
        batchsize = dl.batch_size
        epochs = tqdm(range(epochs), desc=description)
        for epoch in epochs:
            loss = 0
            for batch in dl:
                src, tgt = batch[0], batch[1]

                minLen = min(src.size(1), tgt.size(1))
                src = src[:, :minLen]
                tgt = tgt[:, :minLen]

                logits = self.forward(src, tgt, padId)

                seqLen = tgt.size(1)
                logits = logits[:, :seqLen]

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1)
                )

                lossVal += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lossMessage = f'Loss: {lossVal/batchsize:.4f}'
            epochs.set_description(lossMessage, True)


    def inference(self, srcTokens, sosToken, eosToken, padTokenId, maxLen=120):
        self.eval()
        generated = torch.tensor([[sosToken]], device=srcTokens.device)

        for _ in range(maxLen):
            logits = self.forward(srcTokens, generated, padTokenId)
            nextId = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            generated = torch.cat([generated, nextId], dim=1)

            if nextId.item() == eosToken:
                break

        return generated[0].tolist()

    def addPositionalEmbedding(self, x):
        b, seq = x.size(0), x.size(1)
        pos = torch.arange(seq, device=x.device).unsqueeze(0).expand(b, seq)
        return x + self.positionalEmbedding(pos)

    def causalMask(self, seqLen, device):
        mask = torch.triu(torch.ones(seqLen, seqLen, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def paddingMask(self, tokens, padTokenId):
        return (tokens == padTokenId).unsqueeze(1).unsqueeze(2)

    def decoderMask(self, tokens, padTokenId):
        b, seq = tokens.size()
        pad = self.paddingMask(tokens, padTokenId)
        causal = self.causalMask(seq, tokens.device)
        return pad | causal

    def saveModel(self):
        torch.save(self.state_dict(), "transformerModel.pt")

class EncoderLayer(nn.Module):
    def __init__(self, modelDim, numHeads):
        super().__init__()
        self.selfAttention = MultiHeadAttention(modelDim, numHeads)
        self.mlp = MLP(modelDim)
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)

    def forward(self, x, mask):
        y = x + self.selfAttention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        return y + self.mlp(self.norm2(y))


class DecoderLayer(nn.Module):
    def __init__(self, modelDim, numHeads):
        super().__init__()
        self.selfAttention = MultiHeadAttention(modelDim, numHeads)
        self.crossAttention = MultiHeadAttention(modelDim, numHeads)
        self.mlp = MLP(modelDim)
        self.norm1 = nn.LayerNorm(modelDim)
        self.norm2 = nn.LayerNorm(modelDim)
        self.norm3 = nn.LayerNorm(modelDim)

    def forward(self, x, encOut, selfMask, crossMask):
        y = x + self.selfAttention(self.norm1(x), self.norm1(x), self.norm1(x), selfMask)
        z = y + self.crossAttention(self.norm2(y), encOut, encOut, crossMask)
        return z + self.mlp(self.norm3(z))


class MLP(nn.Module):
    def __init__(self, modelDim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(modelDim, modelDim * 4),
            nn.ReLU(),
            nn.Dropout(0.2)
            nn.Linear(modelDim * 4, modelDim),
            
        )

    def forward(self, x):
        return self.seq(x)
