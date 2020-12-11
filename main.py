import numpy as np
import random
from rnn import RNN
from data2 import train_data, test_data

# Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
size = len(vocab)
print('%d different words are there!' % size)

index = { w: i for i, w in enumerate(vocab) }
word = { i: w for i, w in enumerate(vocab) }

def makeInput(text):
  '''
  Representing the input as a list of 0s and 1s where 
  0 implies that the word is not there in the input and 
  1 implies that the  word is there in the input 
  '''
  input_list = []
  for word in text.split(' '):
    if word not in vocab:
      continue
    voc = np.zeros((size, 1))
    voc[index[word]] = 1
    input_list.append(voc)
  return input_list

def softmax(xs):
  # Uses the Softmax  Activation Function on the input layer.
  return np.exp(xs) / sum(np.exp(xs))

rnn = RNN(size, 2)

def dataProcessing(data, back_propogation=True):
  '''
  Takes the input data sends it to the RNN for forward and back-propogation.
  Also checks the loss and accuracy based on the output of the RNN and returns it.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  correct = 0

  for x, y in items:
    input_list = makeInput(x)
    answer = int(y)
    
    output, _ = rnn.forward(input_list)
    probablity = softmax(output)

    # Calculating the loss vs accuracy score
    loss -= np.log(probablity[answer])
    correct += int(np.argmax(probablity) == answer)

    if back_propogation:
      dL_dy = probablity
      dL_dy[answer] -= 1

      rnn.back_propogation(dL_dy)

  total_loss=loss/len(data)
  acc=correct/len(data)
  return total_loss,acc

# Training loop
for epoch in range(1000):
  train_loss, train_acc = dataProcessing(train_data)

  if epoch % 100 == 0 and epoch!=0:
    print('For Epoch number %d :' % (int(epoch)/100))
    print('Training:\tLoss =  %.3f | Accuracy = %.3f' % (float(train_loss), float(train_acc)))

    test_loss, test_acc = dataProcessing(test_data)
    print('Testing:\tLoss = %.3f | Accuracy = %.3f' % (float(test_loss), float(test_acc)))
