from typing import List

from guacamol.distribution_learning_benchmark import (KLDivBenchmark,
                                                      NoveltyBenchmark,
                                                      UniquenessBenchmark,
                                                      ValidityBenchmark)
from guacamol.distribution_matching_generator import \
    DistributionMatchingGenerator
from guacamol.frechet_benchmark import FrechetBenchmark
from guacamol.utils.chemistry import is_valid

from model.mydataclass import BenchmarkResults


class QuickBenchGenerator(DistributionMatchingGenerator):

    def __init__(self, generator: DistributionMatchingGenerator, number_samples: int=10000, max_tries: int=20):
        self.generator = generator
        
        max_samples = max_tries * number_samples
        number_already_sampled = 0
        number_unique_molecules = 0

        unique_molecules: List[str] = []
        all_molecules: List[str] = []

        iter = 0
        print(f"Starting molecule generation: target {number_samples} unique molecules with max {max_samples} total samples.")
        while number_unique_molecules < number_samples and number_already_sampled < max_samples:
            iter += 1
            remaining_to_sample = number_samples - number_unique_molecules
            print(f"Iteration {iter}: need {remaining_to_sample} more unique molecules.")
            samples = generator.generate(number_samples=remaining_to_sample)
            number_already_sampled += remaining_to_sample
            print(f"Generated {len(samples)} samples. Total samples tried: {number_already_sampled}.")
            for m in samples:
                if is_valid(m):
                    if m not in unique_molecules:
                        unique_molecules.append(m)
                        number_unique_molecules += 1
            all_molecules += samples
            print(f"Current unique molecules count: {number_unique_molecules}.")
        assert len(unique_molecules) >= number_samples

        self.molecules = all_molecules
        self.pt = 0
    
    def generate(self, number_samples: int) -> List[str]:
        samples: List[str] = []
        while len(samples) < number_samples:
            samples.append(self.molecules[self.pt])
            self.pt = (self.pt + 1) % len(self.molecules)
        return samples

class QuickBenchmark(object):

    def __init__(self, training_set: List[str], num_samples: int=10000) -> None:
        self.valid_bench = ValidityBenchmark(number_samples=num_samples)
        self.uniq_bench = UniquenessBenchmark(number_samples=num_samples)
        self.novel_bench = NoveltyBenchmark(number_samples=num_samples, training_set=training_set)
        self.kl_bench = KLDivBenchmark(number_samples=num_samples, training_set=training_set)
        self.fcd_bench = FrechetBenchmark(training_set=training_set, sample_size=num_samples)
        #self.fcd_bench = 0
    def assess_model(self, generator):
        quickbenchgenerator = QuickBenchGenerator(generator)
        valid_result = self.valid_bench.assess_model(quickbenchgenerator)
        uniq_result = self.uniq_bench.assess_model(quickbenchgenerator)
        novel_result = self.novel_bench.assess_model(quickbenchgenerator)
        kl_result = self.kl_bench.assess_model(quickbenchgenerator)
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        fcd_result = self.fcd_bench.assess_model(quickbenchgenerator)
        return BenchmarkResults(
            validity = valid_result,
            uniqueness =  uniq_result,
            novelty = novel_result,
            kl_div = kl_result,
            fcd = fcd_result,
        )