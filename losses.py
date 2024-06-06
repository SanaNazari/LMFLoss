class FocalLoss(nn.Module):
    
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            f'Length of weight tensor must match the number of classes. got {num_classes} expected {len(self.alpha)}'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list,device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        print("LDAM weights:",weight)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.cpu()))
        #m_list = m_list * (max_m / np.max(m_list))
        m_list = (m_list * (max_m / torch.max(m_list))).to(device)
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.device = device
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        #criterion = LDAMLoss(cls_num_list=a list of numer of samples in each class, max_m=0.5, s=30, weight=per_cls_weights)
        """
        max_m: represents the margin used in the loss function. It controls the separation between different classes in the feature space.
        The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
       
        s:  higher s value results in sharper probabilities (more confident predictions), while a lower s value 
        leads to more balanced probabilities. In practice, a larger s value can help stabilize training and improve convergence,
        especially when dealing with difficult optimization landscapes.
        The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """

class LMFLoss(nn.Module):
        def __init__(self,cls_num_list,device,weight,alpha=0.2,beta=0.2, gamma=2, max_m=0.8, s=5,add_LDAM_weigth=False): 
            super().__init__()
            self.focal_loss = FocalLoss(weight, gamma)
            if add_LDAM_weigth:
                LDAM_weight = weight
            else:
                LDAM_weight = None
            print("LMF loss: alpha: ", alpha, " beta: ", beta, " gamma: ", gamma, " max_m: ", max_m, " s: ", s, " LDAM_weight: ", add_LDAM_weigth)
            self.ldam_loss = LDAMLoss(cls_num_list,device, max_m, weight=LDAM_weight, s=s)
            self.alpha= alpha
            self.beta = beta

        def forward(self, output, target):
            focal_loss_output = self.focal_loss(output, target)
            ldam_loss_output = self.ldam_loss(output, target)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss 
