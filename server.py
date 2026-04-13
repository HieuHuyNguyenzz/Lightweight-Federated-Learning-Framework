import torch
from collections import OrderedDict

class Server:
    def __init__(self, model_class, device):
        self.device = device
        self.global_model = model_class().to(device)
        
    def get_global_weights(self):
        return self.global_model.state_dict()

    def aggregate(self, client_weights_list):
        """FedAvg implementation."""
        # Average the weights
        global_dict = OrderedDict()
        num_clients = len(client_weights_list)
        
        for key in client_weights_list[0].keys():
            # Stack tensors and take the mean along the 0th dimension
            stacked_weights = torch.stack([client_weights[key].float() for client_weights in client_weights_list], dim=0)
            global_dict[key] = torch.mean(stacked_weights, dim=0)
            
        self.global_model.load_state_dict(global_dict)
        return global_dict

    def evaluate(self, test_loader):
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return 100. * correct / total
