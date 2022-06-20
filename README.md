# TinyML
TinyML is a blazingly fast, portable `ARM` compatible library for distributed training of ANNs on IoT devices.

It's a tiny, pure **C++** implementation of a Multi-Layer Perceptron, built on a simple linear algebra operations toolkit.

This library comes with an example training on the MNIST dataset, which is included in this repository under `tests/mnist/` folder.

## Usage
In order to run the library locally you need to build the project:
1. `cd tinyML`
2. `cmake .`
3. `cmake --build .` or `make .`

## Example
As an example we train an MLP network to recognize digits from the MNIST dataset:
[main.cpp](https://github.com/ChrisMisan/tinyML/blob/master/tests/src/main.cpp)

## Features
This ANN implementation is meant to be used in edge IoT devices to perform distributed training and/or fine-tuning of predictive models. It utilizes a lightweight binary compression format CBOR to save and read model's weights. 

CBOR's main advantage is the small size of its encoder/decoder implementation in C++, as well as its ubiquitous implementations, making the format readable in all mainstream programming languages, including Java, Python, Javascript, Go, C#, Swift, Dart and many more.

Underlying logic is also simple enough to implement yourself, in case you are not satisfied with existing implementations for your language of choice.

### Federated learning
Given its minuscule size as well as a standardized weight transfer solution this library is a perfect fit for implementing a Federated Learning approach to model training. We were mostly inspired by challenges faced by researchers who authored this paper on [Federated Transfer Learning on Tiny Devices](https://arxiv.org/pdf/2110.01107.pdf), as well as methods mentioned [Comprehensive Survey of Federated Learning for Internet of Things](https://arxiv.org/pdf/2104.07914.pdf).

CBOR lends itself fantastically as a transport medium for model weights, whilst the C++ implementation of linear algebra tools and the Multi-Layer Perceptron should make this process accessible in even the most constrained edge IoT training environment.

This project demonstrates the desired functionality through reading and writing to File Streams. You can easily repurpose the available `save_to_file(...)` and `load_from_file(...)` to save and load network. 

In order test the Federated Learning approach you need to set up your central unit to receive weights from edge devices, combine those and resend them back to all nodes in your IoT network.

## Citations
- _Kopparapu, Kavya & Lin, Eric. (2021). TinyFedTL: Federated Transfer Learning on Tiny Devices._ TinyML has rose to popularity in an era where data is everywhere. However, the data that is in most demand is subject to strict privacy and security guarantees. In addition, the deployment of TinyML hardware in the real world has significant memory and communication constraints that traditional ML fails to address. In light of these challenges, we present TinyFedTL, the first implementation of federated transfer learning on a resource-constrained microcontroller.
- _C. Nguyen, Dinh & Ding, Ming & Pathirana, Pubudu & Seneviratne, Aruna & Li, Jun & Poor, H. Vincent. (2021). Federated Learning for Internet of Things: A Comprehensive Survey. IEEE Communications Surveys & Tutorials. 10.1109/COMST.2021.3075439._ The Internet of Things (IoT) is penetrating many facets of our daily life with the popularity of intelligent services and applications empowered by artificial intelligence (AI). Traditionally, AI techniques require centralized data collection and processing that may not be feasible in realistic application scenarios due to the high scalability of modern IoT networks and growing privacy concerns. Federated Learning (FL) has emerged as a distributed collaborative AI approach that can enable various intelligent IoT applications, by allowing for AI training at distributed IoT devices with user privacy protection. In this article, we provide a comprehensive survey of the emerging applications of FL in IoT networks, beginning from an introduction to the recent advances in FL and IoT to a discussion of their integration. Particularly, we explore and analyze the potential of FL for enabling a wide range of IoT services, including IoT data sharing, data offloading and caching, attack detection, localization, mobile crowdsensing, and IoT privacy and security. We then provide an extensive survey of the use of FL in key IoT applications from different use-case domains such as smart healthcare, smart transportation, Unmanned Aerial Vehicles (UAVs), smart cities, and smart industry. Some important lessons learned from this review of the FL-IoT services and applications are also highlighted. We complete this survey by highlighting current challenges and possible directions for future research in this promising area.
