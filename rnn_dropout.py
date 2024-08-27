import torch
import torch.nn as nn

torch.manual_seed(0xdead)

# for dropout > 0, successive calls to rnn should result in different output since
# nodes are dropped in probabilistic way.
# for dropout is equal to 0, outputs should be same.
for dropout in (0.3, 0):
    rnn = nn.RNN(10, 1000, 2, bias=False, dropout=dropout, nonlinearity='relu')
    input = torch.ones(1, 1, 10)
    output1 = rnn(input)
    output2 = rnn(input)
    print("dropout is ", dropout)
    # print("input is ", input)
    # print("output1 is ", output1)
    # print("output2 is ", output2)
    if dropout == 0 or dropout == 1:
        torch.testing.assert_close(output1, output2)
        print("Elements are close")
    else:
        try:
            torch.testing.assert_close(output1, output2)
        except Exception as e:
            print(e)
            print("Elements are not close")

#o/p
'''
dropout is  0.3
Tensor-likes are not close!

Mismatched elements: 656 / 1000 (65.6%)
Greatest absolute difference: 0.059153834357857704 at index (0, 0, 806) (up to 1e-05 allowed)
Greatest relative difference: inf at index (0, 0, 3) (up to 1.3e-06 allowed)

The failure occurred for item [0]
Elements are not close
dropout is  0
Elements are close
'''
