from src.utils.reproducibility import seed_everything
from src.models.loader import load_model
from src.evaluation.metrics import load_metric
from src.evaluation.results_writer import ResultsWriter
from src.data.loader import load_hf_dataset, group_by_media
from src.procedures.eviction.loader import load_strategy
from config_types.benchmark import BenchmarkConfig



def run_benchmark(config: BenchmarkConfig):
    seed_everything(config.experiment.seed)

    model = load_model(config.model)
    results = ResultsWriter(config.experiment.output_dir)

    for dataset_config in config.datasets:
        # TODO: select correct columns
        dataset = load_hf_dataset(dataset_config)
        dataset = group_by_media(dataset) # TODO

        for strategy in config.eviction.strategies:
            for ratio in config.eviction.ratios:
                # TODO: implement missing presses in kvpress
                args = strategy.args or {}
                eviction_strategy = load_strategy(strategy.name, args | {"compression_ratio": ratio})

                results.write({
                    "dataset": dataset_config.name,
                    "metric": config.scorer,
                    "eviction_strategy": strategy,
                    "eviction_ratio": ratio
                })

                for media_entry in dataset:
                    # Prefill produces the initial model state.
                    # TODO: split entry in media, text. prefill media then query multiple text 
                    media_inputs = model.prepare_inputs(media_entry["media"]) # TODO: correct for positions
                    state = model.prefill(inputs=media_inputs, press=eviction_strategy) # load media

                    for text in media_entry["texts"] : # todo: all questions for one media :
                        # Each trial must have an independent cache state.
                        # cache = create_trial_cache(state)

                        text_inputs = model.prepare_inputs(text) # TODO: correct for positions                    
                        generated_text, output_state = model.predict(state=state, inputs=text_inputs, return_state=True)

                        results.write({
                            "entry_id": text["id"],
                            "output": generated_text,
                            "eviction_indices": output_state.past_key_values, # TODO: just indices
                        })

    results.close()


def calculate_score() :
    # calculate score for one dataset / compression / eviction / ...
    ...