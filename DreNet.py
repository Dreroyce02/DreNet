import random
import math
import pickle

class Neuron:
    def __init__(self, n_of_inputs, activation_func='sigmoid', activation_prime='sigmoid_prime'):
        self.af = activation_func
        self.ap = activation_prime
        self.n = n_of_inputs
        self.W = []
        self.a = 0
        self.z = 0
        self.b = random.random()
        self.dadzdw = []
        self.dadzdi = []
        self.dadzdb = 0
        for _ in range(n_of_inputs):
            self.W.append(random.random())
            self.dadzdw.append(0)
            self.dadzdi.append(0)

    def eval(self, input_):
        try: #incase of overflow
            for _ in range(self.n):
                self.z = input_[_]*self.W[_]
            self.z = self.z + self.b
            if self.af == 'sigmoid':
                self.a = math.pow(math.e, self.z)/(1+math.pow(math.e, self.z))
            else:
                self.a = self.af(self.z)
            if self.ap == 'sigmoid_prime':
                self.dadzdb = math.pow(math.e, self.z)/math.pow(1+math.pow(math.e, self.z),2)
                for _ in range(self.n):
                    self.dadzdw[_] = (math.pow(math.e, self.z)/math.pow(1+math.pow(math.e, self.z),2))*input_[_]
                    self.dadzdi[_] = (math.pow(math.e, self.z)/math.pow(1+math.pow(math.e, self.z),2))*self.W[_]
            else:
                for _ in range(self.n):
                    self.dadzdw[_] = self.ap(self.z)*input_[_]
                    self.dadzdi[_] = self.ap(self.z)*self.W[_]
                self.dadzdb = self.ap(self.z)
        except ArithmeticError:
            self.a = random.random()
            self.z = random.random()
            for _ in range(self.n):
                self.dadzdw[_] = random.random()
                self.dadzdi[_] = random.random()
            self.dadzdb = random.random()
                
class Layer:
    def  __init__(self, neurons):
        self.N = neurons
        self.n = len(neurons)

    def eval(self, _input=[], input_layer=None):
        if _input!=[]:
            for neuron in self.N:
                neuron.eval(_input)
        if input_layer!=None:
            inputs = []
            for neuron in input_layer.N:
                inputs.append(neuron.a)
            for neuron in self.N:
                neuron.eval(inputs)

class Network:
    def __init__(self, layers):
        self.L = layers
        self.n = len(layers)
        self.accuracy = 0

    def eval(self, _input):
        for _ in range(self.n):
            if _==0:
                self.L[_].eval(_input=_input)
            else:
                self.L[_].eval(input_layer=self.L[_-1])
        output_layer = self.L[self.n-1]
        output = []
        for neuron in output_layer.N:
            output.append(neuron.a)
        return output

    def backprop(self, lr, data_set, target_set, loss_func='mse', loss_prime='mse_prime'):
        output_size = self.L[self.n-1].n
        data_set_size = len(data_set)
        loss = 0
        for _ in range(data_set_size):
            inputs = data_set[_]
            output = self.eval(inputs)
            targets = target_set[_]
            dE = []
            for _2 in range(output_size):
                O = output[_2]
                T = targets[_2]
                if loss_prime=='mse_prime':
                    dE.append(2*(O-T)*(1/data_set_size))
                    loss += math.pow(O-T,2)
                else:
                    dE.append(loss_prime(O,T)*(1/data_set_size))
                    loss += abs(loss_func(O,T))
            self.L.reverse()
            for _3 in range(self.n):
                if _3==0:
                    output_layer = self.L[_3]
                    for _4 in range(output_layer.n):
                        dE_o = dE[_4]
                        output_neuron = output_layer.N[_4]
                        for _5 in range(output_neuron.n):
                            output_neuron.W[_5] -= lr*output_neuron.dadzdw[_5]*dE_o
                            output_neuron.b -= lr*dE_o*output_neuron.dadzdb
                            output_neuron.dadzdi[_5] *= dE_o
                else:
                    current_layer = self.L[_3]
                    last_layer = self.L[_3-1]
                    for _4 in range(last_layer.n):
                        output_neuron = last_layer.N[_4]
                        for _5 in range(current_layer.n):
                            input_neuron = current_layer.N[_5]
                            dadzdi = output_neuron.dadzdi[_5]
                            for _6 in range(input_neuron.n):
                                input_neuron.W[_6] -= lr*input_neuron.dadzdw[_6]*dadzdi
                                input_neuron.b -= lr*dadzdi
                                input_neuron.dadzdi[_6] *= dadzdi
            self.L.reverse()
        loss_average = loss/data_set_size
        self.accuracy = (1-loss_average)*100

def gen_layer(n_inputs, n_neurons, act='sigmoid', act_prime='sigmoid_prime'):
    neurons = []
    for _ in range(n_neurons):
        neurons.append(Neuron(n_inputs, activation_func=act, activation_prime=act_prime))
    return Layer(neurons)

def copy(net):
    return Network(net.L)

def save(network, name, location=None):
    if location != None:
        filepath = location+name
    else:
        filepath = name
    with open(filepath, 'ab') as fp:
        pickle.dump(network, fp)

def load(filepath):
    with open(filepath, 'rb') as fp:
        network = pickle.load(fp)
    return network

if __name__=='__main__':
    test_net = Network([gen_layer(4,4),
                        gen_layer(4,2)
                        ])
    
    inputs = []
    targets = []
    for _ in range(100):
        inputs.append([random.random(), random.random(), random.random(), random.random()])
        targets.append([random.random(), random.random()])
    iterations  = 0 
    while test_net.accuracy<80:
        test_net.backprop(.1, inputs, targets)
        iterations += 1
        print(test_net.accuracy)
    print("Number of iterations = {}".format(iterations))
    print(inputs[0])
    print(targets[0])
    print(test_net.eval(inputs[0]))
    copy_net = copy(test_net)
    print(copy_net.eval(inputs[0]))
        
                        
