import torch


class CutLoss(torch.autograd.Function):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    @staticmethod
    def forward(ctx, Y, A):
        ctx.save_for_backward(Y,A)
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        YbyGamma = torch.div(Y, Gamma.t())
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        Y, A, = ctx.saved_tensors
        idx = A._indices()
        data = A._values()
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        # print(Gamma.shape)
        gradient = torch.zeros_like(Y)
        # print(gradient.shape)
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
                            Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
                gradient[i, j] = torch.sum(temp)

                l_idx = list(idx.t())
                l2 = []
                l2_val = []
                # [l2.append(mem) for mem in l_idx if((mem[0] != i).item() and (mem[1] != i).item())]
                for ptr, mem in enumerate(l_idx):
                    if ((mem[0] != i).item() and (mem[1] != i).item()):
                        l2.append(mem)
                        l2_val.append(data[ptr])
                extra_gradient = 0
                if l2:
                    for val, mem in zip(l2_val, l2):
                        extra_gradient += (-D[i] * torch.sum(
                            Y[mem[0], j] * (1 - Y[mem[1], j]) / torch.pow(Gamma[j], 2))) * val

                gradient[i, j] += extra_gradient

        # print(gradient)
        return gradient, None
