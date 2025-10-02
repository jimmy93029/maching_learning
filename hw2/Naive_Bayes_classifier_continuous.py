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

# Function to read the label file (train-labels.idx1-ubyte)
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

# Function to compute MLE (mean, variance) and log probabilities for each pixel, per label
def get_MLE_of_Gaussian(images, labels, image_size=28):
    mean = np.zeros((image_size, image_size, 10), dtype=float)
    variance = np.zeros((image_size, image_size, 10), dtype=float)
    log_prob = np.zeros((image_size, image_size, 10, 256), dtype=float)  # log probabilities for each pixel value (1-256)

    # Iterate over all labels and calculate the mean, variance, and log probabilities
    for label in range(10):
        label_images = images[labels == label]
        
        for row in range(image_size):
            for col in range(image_size):
                pixel_values = label_images[:, row, col]
                mean[row, col, label] = np.mean(pixel_values)
                variance[row, col, label] = np.var(pixel_values) 
                if variance[row, col, label] == 0:
                    variance[row, col, label] = 1e3

                # Compute the log probability for each pixel value (0-255)
                for value in range(256):
                    log_prob[row, col, label, value] = -0.5 * np.log(2 * np.pi * variance[row, col, label]) \
                        - (value - mean[row, col, label])**2 / (2 * variance[row, col, label])

    return mean, variance, log_prob

# Function to predict the label for a test image and return posterior probabilities
def predict_image(image, mean, variance, log_labels, log_prob, image_size=28):
    posterior_probabilities = np.zeros(10)

    for label in range(10):
        prior_prob = log_labels[label]
        likelihood = 0

        for row in range(image_size):
            for col in range(image_size):
                pixel_value = image[row, col]
                likelihood += log_prob[row, col, label, pixel_value]  # Add the precomputed log probability for the pixel

        posterior_probabilities[label] = prior_prob + likelihood

    posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities)
    predicted_label = np.argmin(posterior_probabilities)
    
    return predicted_label, posterior_probabilities

# Function to output results (posterior, imaginations, and predictions)
def output(log_prob_sums, predictions, true_labels, imaginations, filename="continuous_output.txt"):
    with open(filename, 'w') as f:
        # Print out the posterior (in log scale) for each test image
        for i, posterior in enumerate(log_prob_sums):
            f.write(f"Posterior for image {i}:\n")
            for label in range(10):
                f.write(f"{label}: {posterior[label]:.16f}\n")
            f.write(f"Prediction: {predictions[i]}, True Label: {true_labels[i]}\n\n")

        # Output the imaginations of numbers for each label (0-9)
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

# Function to create imaginations of numbers by choosing the most likely pixel (0 or 1)
def create_imaginations(mean, variance, image_size=28):
    imaginations = []
    for label in range(10):
        imagination = np.zeros((image_size, image_size), dtype=int)

        for row in range(image_size):
            for col in range(image_size):
                pixel_mean = mean[row, col, label]
                pixel_variance = variance[row, col, label]

                # Use the pixel value with the highest probability (0 if pixel value < 128, 1 otherwise)
                most_probable_pixel = int(round(pixel_mean))  # Choose the mean value for most probable
                imagination[row, col] = 0 if most_probable_pixel < 128 else 1  # 0 for white, 1 for black

        imaginations.append(imagination)
    return imaginations

# Main function to run the classifier
def main():
    train_images = read_idx_images('train-images.idx3-ubyte_')
    train_labels = read_idx_labels('train-labels.idx1-ubyte_')
    test_images = read_idx_images('t10k-images.idx3-ubyte_')
    test_labels = read_idx_labels('t10k-labels.idx1-ubyte_')

    # Compute MLE of mean, variance, and log probabilities for Gaussian distributions
    mean, variance, log_prob = get_MLE_of_Gaussian(train_images, train_labels)

    # Compute prior probabilities (P(y)) for each label
    label_counts = np.bincount(train_labels)
    log_labels = np.log(label_counts / len(train_labels))

    posteriors = []
    predictions = []
    imaginations = create_imaginations(mean, variance)

    # Process each test image
    for test_image in test_images:
        predicted_label, posterior_probabilities = predict_image(test_image, mean, variance, log_labels, log_prob)
        predictions.append(predicted_label)  # Store the predicted label
        posteriors.append(posterior_probabilities)  # Store the posterior for this image

    # Output the results and imaginations
    output(posteriors, predictions, test_labels, imaginations)

if __name__ == '__main__':
    # Run the program
    main()
