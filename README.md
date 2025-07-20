Nesse projeto pude desenvolver minhas habilidades em visão computacional e em python.

O modelo conta com um repositório de treinamento, um de validação e um de input. O reconhecimento facial não é em tempo real, 
mas isso pode ser facilmente adaptado.
Outra particularidade é que o treinamento do dataset não foi salvo, sendo treinado sempre que o código é executado, mas é algo também alterável.

*Algumas bibliotecas utilizadas:
face_recognition --> Usada para o reconhecimento facial;

PIL --> Auxilia no desenho do Bounding Box;

collections --> Usa a classe Counter como um esquema de votação para definir
o nome mais compatível de acordo com as codificações do
dataset;

pathlib --> Usa a classe Path na manipulação dos caminhos de arquivos e de
diretórios.
