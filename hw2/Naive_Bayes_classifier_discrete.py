import numpy as np

# Function to read the image file (train-images-idx3-ubyte)
def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        if magic_number != 2051:
            raise ValueError("This is not a valid image file.")
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_columns = int.from_bytes(f.read(4), byteorder='big')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_columns)
        return images

# Function to read the label file (train-labels-idx1-ubyte)
def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        if magic_number != 2049:
            raise ValueError("This is not a valid label file.")
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        if len(labels) != num_labels:
            raise ValueError(f"Expected {num_labels} labels, but found {len(labels)}.")
        return labels

# Function to bin pixel values into 32 bins
def bin_pixels(image):
    return image // 8

# Function to process the labels and collect frequency in bins
def classify_pixels(images, labels, image_size=28, bin_size=32):
    # Create a 4D array with shape (28, 28, bins, 10) to store label counts for each bin
    bin_counts = np.zeros((image_size, image_size, bin_size, 10), dtype=float)
    label_counts = np.zeros(10, dtype=int)

    # Iterate over all images and their labels
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]

        # Bin the pixel values
        binned_image = bin_pixels(image)

        # Update bin counts for the current image's pixels
        for row in range(image_size):
            for col in range(image_size):
                bin_index = binned_image[row, col]  # Get the bin for this pixel
                bin_counts[row, col, bin_index, label] += 1
        label_counts[label] += 1

    return bin_counts, label_counts


# Function to calculate log probabilities P(label | bin_index, row, col)
def make_log_prob(bin_counts, label_counts, image_size=28, bin_size=32):
    log_probs = np.zeros((image_size, image_size, bin_size, 10), dtype=float) 

    # Compute log probabilities for each bin and label
    for row in range(image_size):
        for col in range(image_size):
            for label in range(10):
                total_count = bin_counts[row, col, :, label].sum()  # Total count for this label across all bins
                for bin_index in range(bin_size):
                    # Probability P(bin | label)
                    prob = bin_counts[row, col, bin_index, label] / total_count
                    log_probs[row, col, bin_index, label] = np.log(max(1e-4, prob))

    # Log prior probabilities for labels
    log_labels = np.log(label_counts / label_counts.sum())

    return log_probs, log_labels


# Function to predict the label for a test image and return posterior probabilities
def predict_image(image, log_probs, log_labels, image_size=28):
    # Convert the test image's pixel values to bins
    binned_image = bin_pixels(image)

    # Initialize an array to store the log probability summation for each label
    log_prob_sum = np.zeros(10)

    # Iterate through each pixel in the image (28x28 pixels)
    for row in range(image_size):
        for col in range(image_size):
            bin_index = binned_image[row, col]  # Get the bin for the pixel

            # For each label, sum the log probabilities of the corresponding bin
            for label in range(10):
                log_prob_sum[label] += log_probs[row, col, bin_index, label]

    # Add the log prior probabilities for each label
    log_prob_sum += log_labels

    posterior = log_prob_sum / np.sum(log_prob_sum)
    prediction = np.argmin(posterior)

    return posterior, prediction


# Function to create the "imagination" image based on log probabilities
def create_imagination(label, log_probs, image_size=28, bin_size=32):
    imagination = np.zeros((image_size, image_size), dtype=int)

    # Loop through each pixel in the image
    for row in range(image_size):
        for col in range(image_size):
            # Find the bin with the highest probability for this pixel and label
            bin_index = np.argmax(log_probs[row, col, :, label])  # Get the bin with the highest log-probability

            # Decide if it's white or black based on the bin with the highest probability
            if bin_index < 16:
                imagination[row, col] = 0  # White pixel (lower bins)
            else:
                imagination[row, col] = 1  # Black pixel (higher bins)

    return imagination


# Function to output results (posterior, imaginations, and predictions)
def output(log_prob_sums, imaginations, predictions, true_labels, filename="discrete_output.txt"):
    with open(filename, 'w') as f:
        # Print out the posterior (in log scale) for each test image
        for i, posterior in enumerate(log_prob_sums):
            f.write(f"Posterior (in log scale) for image {i}:\n")
            for label in range(10):
                f.write(f"{label}: {posterior[label]:.16f}\n")
            f.write(f"Prediction: {predictions[i]}, True Label: {true_labels[i]}\n\n")


        # Print out the imaginations of numbers for each label (0-9)
        f.write("Imagination of numbers by Bayesian classifier:\n")
        for label in range(10):
            f.write(f"Label {label}:\n")
            for row in imaginations[label]:
                f.write(' '.join(str(x) for x in row) + '\n')
            f.write("\n")

        # Calculate error rate
        error_count = np.sum(np.array(predictions) != np.array(true_labels))
        error_rate = error_count / len(true_labels)
        f.write(f"Error rate: {error_rate:.4f}\n")


# Main function to run the classifier
def main():
    train_images = read_idx_images('train-images.idx3-ubyte_')
    train_labels = read_idx_labels('train-labels.idx1-ubyte_')
    test_images = read_idx_images('t10k-images.idx3-ubyte_')
    test_labels = read_idx_labels('t10k-labels.idx1-ubyte_')

    # Classify pixels and calculate log probabilities
    bin_counts, label_counts = classify_pixels(train_images, train_labels)
    log_probs, log_labels = make_log_prob(bin_counts, label_counts)

    posteriors = []
    imaginations = []
    predictions = []

    # Process each test image
    for test_image in test_images:
        posterior, prediction = predict_image(test_image, log_probs, log_labels)
        predictions.append(prediction)  # Store the predicted label
        posteriors.append(posterior)  # Store the log posterior

    # Generate imaginations for each label (0-9)
    for label in range(10):
        imaginations.append(create_imagination(label, log_probs))

    # Output the results
    output(posteriors, imaginations, predictions, test_labels)


if __name__ == '__main__':
    # Run the program
    main()
