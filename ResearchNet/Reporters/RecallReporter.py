import torch

def report_recall(network, testing_set, name, fold):
	correct = 0
	total = 0
	true_healthy = 0
	true_leuko = 0
	total_healthy = 0
	total_leuko = 0
	# do testing without gradient because we don't want to learn from these
	with torch.no_grad():
		for i in range(len(testing_set['images'])//100+1):
			images = testing_set['images'][i*100:(i+1)*100]
			labels = testing_set['labels'][i*100:(i+1)*100]
			outputs = network(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			true_healthy += ((predicted == 0) & (labels == 0)).sum().item()
			true_leuko += ((predicted == 1) & (labels == 1)).sum().item()
			total_healthy += (labels == 0).sum().item()
			total_leuko += (labels == 1).sum().item()
	print('Accuracy of the network on the {} test images {}'.format(total, correct/total))
	print('True healthy {}/{}'.format(true_healthy, total_healthy))
	print('True Leuko {}/{}'.format(true_leuko, total_leuko))
	with open(name, 'a') as open_file:
		open_file.write('Fold:'+str(fold)+'\n')
		open_file.write(str(correct/total)+'\n')
		open_file.write(str(true_healthy)+'\n')
		open_file.write(str(total_healthy)+'\n')
		open_file.write(str(true_leuko)+'\n')
		open_file.write(str(total_leuko)+'\n')

