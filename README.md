# Master-Thesis

ai-project/
├── pyproject.toml
├── README.md
├── configs/
│ ├── models/
│ │ ├── resnet50.yaml
│ │ └── vit.yaml
│ ├── datasets/
│ │ └── imagenet.yaml
│ ├── experiments/
│ │ └── baseline.yaml
│ └── benchmarks/
│ └── latency.yaml
│
├── src/
│ └── ai_project/
│ ├── models/
│ │ ├── **init**.py
│ │ ├── base.py
│ │ ├── registry.py
│ │ ├── resnet.py
│ │ ├── vit.py
│ │ └── custom/
│ │ ├── architecture.py
│ │ └── procedures.py
│ │
│ ├── data/
│ │ ├── datasets.py
│ │ ├── loaders.py
│ │ ├── transforms.py
│ │ └── splits.py
│ │
│ ├── training/
│ │ ├── trainer.py
│ │ ├── losses.py
│ │ ├── optimizers.py
│ │ └── callbacks.py
│ │
│ ├── inference/
│ │ ├── engine.py
│ │ ├── pipeline.py
│ │ └── postprocessing.py
│ │
│ ├── evaluation/
│ │ ├── metrics.py
│ │ └── evaluators.py
│ │
│ ├── benchmarking/
│ │ ├── latency.py
│ │ ├── throughput.py
│ │ ├── memory.py
│ │ └── runner.py
│ │
│ ├── procedures/
│ │ ├── base.py
│ │ └── custom_method.py
│ │
│ └── utils/
│ ├── logging.py
│ ├── reproducibility.py
│ └── checkpointing.py
│
├── scripts/
│ ├── train.py
│ ├── infer.py
│ ├── evaluate.py
│ └── benchmark.py
│
├── tests/
│ ├── test_models.py
│ ├── test_data.py
│ └── test_inference.py
│
├── experiments/
│ └── README.md
│
└── outputs/
├── checkpoints/
├── predictions/
├── benchmarks/
└── logs/
