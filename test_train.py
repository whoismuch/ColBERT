import torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Trainer

if __name__=='__main__':
    print(torch.backends.mps.is_available())

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=8,
            root="./exp"
        )
        trainer = Trainer(
            triples="xaa.json",
            queries="queries.tsv",
            collection="collection-cutted.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint='colbert-ir/colbertv1.9', )

        print(f"Saved checkpoint to {checkpoint_path}...")