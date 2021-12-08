# Federated-Learning-with-MPI
This is the Parallel Programming Project using MPI4PY in the Federated Learning Algorithm concept. 

In the centralized federated learning setting, a central server is used to orchestrate the different steps of the algorithms and coordinate all the participating nodes during the learning process. The server is responsible for the nodes selection at the beginning of the training process and for the aggregation of the received model updates. Since all the selected nodes have to send updates to a single entity, the server may become a bottleneck of the system.
![Federated_learning_process_central_case](https://user-images.githubusercontent.com/53828158/145125741-a438e4cf-2519-476b-88f1-316c0df14aac.png)

In the decentralized federated learning setting, the nodes are able to coordinate themselves to obtain the global model. This setup prevents single point failures as the model updates are exchanged only between interconnected nodes without the orchestration of the central server. Nevertheless, the specific network topology may affect the performances of the learning process. 

![Decentralized-Federated-Learning-Environments](https://user-images.githubusercontent.com/53828158/145125956-70d59c9e-6f66-4dee-8b8f-d0a1d592f79f.png)
