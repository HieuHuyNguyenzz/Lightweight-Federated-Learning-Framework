import torch
from collections import OrderedDict

class FLServer:
    """Generic FL Server that manages the global model and aggregation strategy."""
    
    def __init__(self, model_class, strategy, config):
        self.config = config
        self.device = config.device
        self.strategy = strategy
        self.global_model = model_class().to(self.device)
        
    def get_global_weights(self):
        return self.global_model.state_dict()

    def aggregate(self, client_weights_list):
        """Aggregates weights using the injected strategy."""
        aggregated_weights = self.strategy.aggregate(client_weights_list)
        
        if aggregated_weights:
            self.global_model.load_state_dict(aggregated_weights)
            
        return aggregated_weights

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
