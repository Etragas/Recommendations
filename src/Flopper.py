class Flopper():
    flop_count = 0


    def flopMean(self,x):
        flop_count += 5
        return torch.mean(x)

    def flopForward(self, x, network):
        for layer in network:
            flop_count += some_logic

        return network.forward(x)

