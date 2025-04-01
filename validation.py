import torch
#import other files here

#pulling model from file
#model.eval()

#random values for testing
validation_loss = 0.01
validation_correct = 0.01
validation_total = 0.01

#----------------
#torch code here
#----------------

#validation printing
validation_accuracy = validation_correct / validation_total * 100
print(f" Validation Accuracy: {validation_accuracy}")