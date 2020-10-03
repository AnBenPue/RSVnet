import json
import matplotlib.pyplot as plt
# Path to the json file with the metrics
base_path = '/home/ant69/RSVnet/RSVnet_global/models/'
model_name = 'myRSVnet_metrics.json'
# Open the file and load the data
with open(str(base_path+model_name), "r") as read_file:
    metrics_data = json.load(read_file)
# Accuracy and loss data for training and test set
train_acc = metrics_data['training']['accuracy']
test_acc = metrics_data['test']['accuracy']
train_loss = metrics_data['training']['loss']
test_loss = metrics_data['test']['loss']
# Build the plot object
plt.subplot(211)
plt.plot(train_acc,label='train')
plt.plot(test_acc, label='test')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend(loc="upper right")

plt.subplot(212)
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.ylabel('loss')
plt.title('Loss')
plt.legend(loc="upper right")
plt.show()