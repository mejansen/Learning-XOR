from MLP import *
from LogicOperators import *

import random



def generatorLogicFunction(operator):
    bools = [True, False]
    while True:
        x1 = bools[random.randint(0, 1)]
        x2 = bools[random.randint(0, 1)]
        target = operator.result(x1, x2)

        result = [x1, x2, target]

        # Bool to [0,1]
        for i, value in enumerate(result):
            if value:
                result[i] = 1
            else:
                result[i] = 0

        yield result


def getPerformance(mlp, operator, testSize=50):
    THRESHOLD = 0.5
    loss = 0
    cntCorrect = 0
    for i in range(testSize):
        data = next(generatorLogicFunction(operator))
        [x1, x2, target] = data
        y = mlp.forward_step([data[0], data[1]])
        y = y[0]  # extract value
        loss += (y - target) ** 2

        # MLP said "this is true"
        if y > THRESHOLD:
            # correct?
            if target == 1:
                cntCorrect += 1

        # MLP said "this is false"
        else:
            # correct?
            if target == 0:
                cntCorrect += 1

    accuracy = cntCorrect / testSize
    avgLoss = loss / testSize
    return accuracy, avgLoss


def main():
    epsilon = 1  # Learning rate
    NUM_EPOCHS = 1000
    operators = [XOR()]

    row_idx = 0
    column_idx = 0
    for operator in operators:
        mlp = MLP([2, 5, 1])


        performance = []
        loss = []
        for i in range(NUM_EPOCHS):
            # Measure
            avgPerformance, avgLoss = getPerformance(mlp, operator)
            performance.append(avgPerformance)
            loss.append(avgLoss)

            # Train
            data = next(generatorLogicFunction(operator))
            [x1, x2, target] = data
            mlp.backprop_step([x1, x2], target, epsilon)

            print(f"Average Performance: {avgPerformance}; Average Loss: {avgLoss}")


    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
