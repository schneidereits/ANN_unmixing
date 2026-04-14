################################################################################
#                                  Import                                      #
################################################################################
import numpy as np
import random
import os
from tqdm import tqdm
from prm import DATA_DIR, SYNTHMIX_DIR, CLASSES, NUMBER_OF_SAMPLES, CLASS_PROBABILITIES, MIXING_COMPLEXITY_PROBABILITIES, INCL_PURE_LIBRARY, EQUALIZE_SAMPLES
from prm import *

################################################################################
#                                  User Settings                               #
################################################################################
# input
input_directory = DATA_DIR
input_file_names = [f"{c}.csv" for c in CLASSES]
class_names = CLASSES

# output
output_directory = SYNTHMIX_DIR
os.makedirs(output_directory, exist_ok=True)
output_file_name_spec = 'mixed_spectra.npy'
output_file_name_frac = 'fraction_label.npy'

# synthetic mixing parameters
number_of_samples = NUMBER_OF_SAMPLES
class_probabilities = CLASS_PROBABILITIES
mixing_complexity_probabilities = MIXING_COMPLEXITY_PROBABILITIES
incl_pure_library = INCL_PURE_LIBRARY


################################################################################
#                           Function Definitions                               #
################################################################################

def synth_mixing(input_directory, output_directory, input_file_names, class_names, output_file_name_spec, output_file_name_frac, number_of_samples, class_probabilities=None, mixing_complexity_probabilities=None, incl_pure_library=False, EQUALIZE_SAMPLES=EQUALIZE_SAMPLES):

    # Create library dictionary (class names & library files)
    library_dict = {
        class_name: np.genfromtxt(os.path.join(input_directory, input_file_names), delimiter=',')
        for class_name, input_file_names in zip(class_names, input_file_names)}

    # Number of classes
    num_classes = len(class_names)

    # Check and set class probabilities (default is equalized if not provided)
    if class_probabilities is None:
        class_probabilities = [1 / num_classes] * num_classes  # Equal probabilities if not defined
    else:
        if sum(class_probabilities) != 1.0:
            raise ValueError(
                "Warning: Class probabilities must sum to 1. Current sum: {:.2f}".format(sum(class_probabilities)))

    # Check mixing complexity probabilities
    if mixing_complexity_probabilities is None:
        raise ValueError("Error: Mixing complexity probabilities must be provided.")
    if sum(mixing_complexity_probabilities) != 1.0:
        raise ValueError("Warning: Mixing complexity probabilities must sum to 1. Current sum: {:.2f}".format(
            sum(mixing_complexity_probabilities)))

    # Initialize lists for synthetic training data
    mixed_spectral_data = []
    label_data = []

    # Create synthetic mixtures
    for _ in tqdm(range(number_of_samples)):

        # Initialize spectral mixture (x) and fractions for each class (y)
        x = 0
        y = np.zeros(len(class_names), np.float32)

        # Randomly determine mixing complexity
        mixing_complexity = random.choices(range(1, len(mixing_complexity_probabilities) + 1), weights=mixing_complexity_probabilities, k=1)[0]

        # Randomly determine classes from which spectra will be pulled
        selected_classes = random.choices(class_names, k=mixing_complexity, weights=class_probabilities)

        # Create random fractions for the classes from which spectra will be pulled
        random_fraction = np.random.dirichlet(np.ones(len(selected_classes)), size=1)[0]

        # Create the synthetic spectral mixture
        for i in range(len(selected_classes)):
            # Index for pulling a random spectrum from the current class
            cur_spec_index = random.randrange(library_dict[selected_classes[i]].shape[0])
            cur_spectrum = library_dict[selected_classes[i]][cur_spec_index]

            # calculate the mixture and integrate it into the total mixture
            x += cur_spectrum * random_fraction[i]

            # add the fraction to the label data
            cur_frac_index = class_names.index(selected_classes[i])
            y[cur_frac_index] += random_fraction[i]

        # Add the mixed spectral sample and its label to the dataset
        mixed_spectral_data.append(x)
        label_data.append(y)

    # Append input library to the mixed_spectral_data and label_data
    if incl_pure_library:
        for class_name in class_names:
            spectra = library_dict[class_name]
            num_spectra = spectra.shape[0]
            mixed_spectral_data = np.vstack((mixed_spectral_data, spectra))
            label_for_class = np.zeros((num_spectra, len(class_names)), dtype=np.float32)
            label_for_class[:, class_names.index(class_name)] = 1  # Set fraction 1 for the current class
            label_data = np.vstack((label_data, label_for_class))

    # Convert to array
    mixed_spectral_data = np.asarray(mixed_spectral_data, dtype=np.float32)
    label_data = np.asarray(label_data, dtype=np.float32)


    # Print the histogram counts for each class
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Define bins explicitly
    counts = {class_name: np.zeros(len(bins), dtype=int) for class_name in class_names}
    for y in label_data:
        for class_index, class_name in enumerate(class_names):
            if y[class_index] == 0.0:
                counts[class_name][0] += 1  # Count for the bin 0.0
            elif y[class_index] == 1.0:
                counts[class_name][-1] += 1  # Count for the bin 1.0
            else:
                bin_index = np.digitize(y[class_index], bins) - 1
                if 0 < bin_index < len(counts[class_name]) - 1:  # Ensure we're within valid range
                    counts[class_name][bin_index] += 1  # Increment for <0.1, <0.2, ..., <0.9
    for class_name in class_names:
        print(f'Histogram for class "{class_name}":')
        print(f'  0.0: {counts[class_name][0]}')  # Count for bin 0.0
        for bin_index in range(1, len(bins) - 1):
            print(f'  <{bins[bin_index]}: {counts[class_name][bin_index]}')  # Count for bins <0.1, <0.2, ..., <0.9
        print(f'  {bins[-1]}: {counts[class_name][-1]}')  # Count for bin 1.0

################################################################################
    counts[class_name][bin_index] += 1  # Increment for <0.1, <0.2, ..., <0.9
    for class_name in class_names:
        print(f'Histogram for class "{class_name}":')
        print(f'  0.0: {counts[class_name][0]}')  # Count for bin 0.0
        for bin_index in range(1, len(bins) - 1):
            print(f'  <{bins[bin_index]}: {counts[class_name][bin_index]}')  # Count for bins <0.1, <0.2, ..., <0.9
        print(f'  {bins[-1]}: {counts[class_name][-1]}')  # Count for bin 1.0

        # --------------------------------------------------------------------------
    # Optional class equalization by bin count
    # --------------------------------------------------------------------------
    if EQUALIZE_SAMPLES:
        print("\n=== Equalizing class bins ===")

        # Collect sample indices for each (class, bin)
        bin_indices = {c: {b: [] for b in range(len(bins))} for c in class_names}
        for idx, y in enumerate(label_data):
            for ci, cname in enumerate(class_names):
                val = y[ci]
                if val == 0.0:
                    b = 0
                elif val == 1.0:
                    b = len(bins) - 1
                else:
                    b = np.digitize(val, bins) - 1
                    if not (0 < b < len(bins) - 1):
                        continue
                bin_indices[cname][b].append(idx)

        # Find minimum non-empty bin count
        min_count = min(
            len(idxs)
            for cdict in bin_indices.values()
            for idxs in cdict.values()
            if len(idxs) > 0
        )
        print(f"Equalizing each (class, bin) to {min_count} samples.")

        selected_indices = []
        for cname in class_names:
            for b, idxs in bin_indices[cname].items():
                if len(idxs) >= min_count:
                    selected_indices.extend(random.sample(idxs, min_count))
                # skip empty bins

        selected_indices = sorted(selected_indices)
        mixed_spectral_data = mixed_spectral_data[selected_indices]
        label_data = label_data[selected_indices]
        print(f"After equalization: {len(selected_indices)} total samples retained.")
        
        # Print the histogram counts for each class
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Define bins explicitly
    counts = {class_name: np.zeros(len(bins), dtype=int) for class_name in class_names}
    for y in label_data:
        for class_index, class_name in enumerate(class_names):
            if y[class_index] == 0.0:
                counts[class_name][0] += 1  # Count for the bin 0.0
            elif y[class_index] == 1.0:
                counts[class_name][-1] += 1  # Count for the bin 1.0
            else:
                bin_index = np.digitize(y[class_index], bins) - 1
                if 0 < bin_index < len(counts[class_name]) - 1:  # Ensure we're within valid range
                    counts[class_name][bin_index] += 1  # Increment for <0.1, <0.2, ..., <0.9
    for class_name in class_names:
        print(f'Histogram for class "{class_name}":')
        print(f'  0.0: {counts[class_name][0]}')  # Count for bin 0.0
        for bin_index in range(1, len(bins) - 1):
            print(f'  <{bins[bin_index]}: {counts[class_name][bin_index]}')  # Count for bins <0.1, <0.2, ..., <0.9
        print(f'  {bins[-1]}: {counts[class_name][-1]}')  # Count for bin 1.0

################################################################################
    counts[class_name][bin_index] += 1  # Increment for <0.1, <0.2, ..., <0.9
    for class_name in class_names:
        print(f'Histogram for class "{class_name}":')
        print(f'  0.0: {counts[class_name][0]}')  # Count for bin 0.0
        for bin_index in range(1, len(bins) - 1):
            print(f'  <{bins[bin_index]}: {counts[class_name][bin_index]}')  # Count for bins <0.1, <0.2, ..., <0.9
        print(f'  {bins[-1]}: {counts[class_name][-1]}')  # Count for bin 1.0
        
    # Save synthetic mixtures and labels to .npy files
      # Convert to array
    mixed_spectral_data = np.asarray(mixed_spectral_data, dtype=np.float32)
    label_data = np.asarray(label_data, dtype=np.float32)

    # Save
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, output_file_name_spec), arr=mixed_spectral_data)
    np.save(os.path.join(output_directory, output_file_name_frac), arr=label_data)


################################################################################
#                               Execution                                      #
################################################################################

def main():
    print('\n=== Script Execution Started ===')

    synth_mixing(
        input_directory,
        output_directory,
        input_file_names,
        class_names,
        output_file_name_spec,
        output_file_name_frac,
        number_of_samples,
        class_probabilities,
        mixing_complexity_probabilities,
        incl_pure_library
    )

    print('\n=== Script Execution Completed ===')

if __name__ == '__main__':
    main()
