# Federated-Learning-with-MPI
This is the Parallel Programming Project using MPI4PY in the Federated Learning Algorithm concept. 

In the centralized federated learning setting, a central server is used to orchestrate the different steps of the algorithms and coordinate all the participating nodes during the learning process. The server is responsible for the nodes selection at the beginning of the training process and for the aggregation of the received model updates. Since all the selected nodes have to send updates to a single entity, the server may become a bottleneck of the system.
![Federated_learning_process_central_case](https://user-images.githubusercontent.com/53828158/145125741-a438e4cf-2519-476b-88f1-316c0df14aac.png)

# Idea of Centralized Federated Learning In Parallel

1. The Master process read a data set.

2. The Master process will scatter all dataset divided by number of processes to each process.

3. Master process will broadcast the newest weight of model to each process.

4. Each process produces updates to the model parameters based on their individual datasets.

5. Each process then send their client-model updates back to the Master.

6. The Master process aggregates the updates to produce an updated version of the global model.

7. The Master process then broadcast the new version to all processes, and the entire pro- cess repeats itself until convergence.

<img width="567" alt="Screen Shot 2022-08-04 at 3 52 45 PM" src="https://user-images.githubusercontent.com/53828158/182864234-8fef7cf1-e688-401d-8064-8af702b34ba3.png">

## Tasks in Concurrency
+ 3rd Task: Each Process produces updates to the model parameters based on their indi- vidual datasets.

+ 4th Task: The processes then send their client-model updates back to the central server (Master process).

## Tasks in Sequence:
+ 1st Task: Master process splits data based on division of the amount of data to the number of processes.

+ 2nd Task: Scatter data and broadcast data to each process before doing parallel.

+ 5th Task: Master process aggregates the updates to produce an updated version of the global model.

+ 6th Task: The Master process then broadcast the new version of weight to the client devices, and the entire steps repeat itself until convergence.

# Installation
1. Clone repo
    ```bash
    git clone https://github.com/dungdao1995/Federated-Learning-with-MPI.git
    ```
2. Install requirements
      ```bash
   pip install -r requirements.txt
    ```
3. Run the Sequential model

     ```bash
   python Centralized_sequential.py
    ```
4. Run the Parallel model with MPI
     ```bash
   mpirun -n 2 python Centralized_MPI.py
    ```

# Result 
In academy purpose, I used 2 processors in MPI for running Centralized Federated Leanrning . The device I used the Macbook pro 2015 - 13 inch with con figuration:
CPU: 2.7GHz dual-core Intel Core i5 processor
RAM: 8GB of 1866MHz LPDDR3 onboard memory
Operation System: MacOS


<img width="591" alt="Screen Shot 2021-12-12 at 7 03 50 PM" src="https://user-images.githubusercontent.com/53828158/145724050-45daf822-6652-4b3a-b05f-ea47b4dc9f6b.png">
