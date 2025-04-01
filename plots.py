import matplotlib.pyplot as plt

#Variables that will be defined in other files
#random values for testing graph
training_loss = [0.573179, 0.510438, 0.479182, 0.462226]
validation_loss = [0.576127, 0.510341, 0.479574, 0.464549]
test_loss = [0.573392, 0.514302, 0.483426, 0.463203]

#Loss Plot------------------------------------
plt.plot(training_loss, label='training Loss', color='blue')
plt.plot(validation_loss, label='validation Loss', color='green')
plt.plot(test_loss, label='Test Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')

plt.legend()
plt.grid(True)
plt.show()

#------------------------------------
