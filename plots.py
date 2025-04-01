import matplotlib.pyplot as plt

#Variables that will be defined in other files
#random values for testing graph
training_loss = [0.573179, 0.510438, 0.479182, 0.462226]
validation_loss = [0.576127, 0.510341, 0.479574, 0.464549]
test_loss = [0.573392, 0.514302, 0.483426, 0.463203]

training_accuracy = [79.8500, 82.0300, 83.0300, 83.6100]
validation_accuracy = [80.1200, 82.1100, 83.0600, 83.1900]
test_accuracy = [80.0700, 81.9900, 82.7500, 83.6100]

epochs = [25, 40, 55, 70]

#Loss Plot----------------------------------------
plt.plot(epochs, training_loss, label='training Loss', color='blue')
plt.plot(epochs, validation_loss, label='validation Loss', color='green')
plt.plot(epochs, test_loss, label='Test Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')

plt.xticks(range(min(epochs), max(epochs) + 1, 5))
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------
plt.clf()
#Accuracy Plot------------------------------------
plt.plot(epochs, training_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', color='green')
plt.plot(epochs, test_accuracy, label='Test Accuracy', color='red')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')

plt.xticks(range(min(epochs), max(epochs) + 1, 5))
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------