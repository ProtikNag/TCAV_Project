import os
from config import (
    GRAPH_PATH, LABEL_PATH, WORKING_DIR, ACTIVATION_DIR, CAV_DIR,
    TARGET_CLASS, CONCEPTS, BOTTLENECKS, ALPHAS, NUM_RANDOM_EXP
)
from models.google_net_wrapper import GoogleNetWrapper
from tcav_core.activation_generator import ImageActivationGenerator
from tcav_core.tcav_analysis import TCAVAnalysis
from plots import plot_results


def main():
    # Load model
    print("Initializing the model...")
    model = GoogleNetWrapper(GRAPH_PATH, LABEL_PATH)

    # Initialize activation generator
    print("Setting up the activation generator...")
    act_generator = ImageActivationGenerator(
        model=model,
        source_dir=os.path.join(WORKING_DIR, 'data'),
        activation_dir=ACTIVATION_DIR
    )

    # Run TCAV analysis
    print("Starting TCAV analysis...")
    tcav_analysis = TCAVAnalysis(
        model=model,
        target=TARGET_CLASS,
        concepts=CONCEPTS,
        bottlenecks=BOTTLENECKS,
        act_generator=act_generator,
        alphas=ALPHAS,
        cav_dir=CAV_DIR,
        num_random_exp=NUM_RANDOM_EXP
    )

    results = tcav_analysis.run(run_parallel=False)
    print("TCAV analysis completed.")

    # Plot results
    print("Plotting results...")
    plot_results.plot_results(results, num_random_exp=NUM_RANDOM_EXP)

    # Close the model session
    model.close()


if __name__ == "__main__":
    main()
