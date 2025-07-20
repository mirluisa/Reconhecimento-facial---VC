from collections import Counter
from pathlib import Path
import time

import face_recognition
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------------------------------

COR_BOUNDING_BOX = "green"   # Cor do bounding box
COR_TEXTO = "white"          # Cor do texto

dataset = {}                 # Dicionario com os nomes e codificacoes de face

# --------------------------------------------------------------------------------------------------


def treinamento():
    """
    Realiza o treinamento do modelo, armazenando as codificacoes e identificacoes das faces no
    dataset.
    """

    print("\n[TREINAMENTO] Treinando modelo...")

    inicio = time.time()

    global dataset       # Obtem a referencia da variavel dataset
    nomes = []           # Nomes dos atores
    codificacoes = []    # Codificacoes das imagens

    # Percorre os diretórios de treinamento
    for imagem_treinamento in Path("treinamento").glob("*/*"):

        # Obtem o nome do diretorio e carrega as imagens
        nome = imagem_treinamento.parent.name
        imagem = face_recognition.load_image_file(imagem_treinamento)

        # Obtem as coordenadas do bounding box da face localizada
        localizacao_face = face_recognition.face_locations(imagem)

        # Obtem a codificacao da face localizada
        codificacao_face = face_recognition.face_encodings(imagem, localizacao_face)

        # Salva o nome e codificacoes da face localizada
        for codificacao in codificacao_face:
            nomes.append(nome)
            codificacoes.append(codificacao)

    # Salva os valores obtidos no dicionario de dataset
    dataset = {"nomes": nomes, "codificacoes": codificacoes}

    fim = time.time()

    print(f"\n[TREINAMENTO] Treinamento finalizado. (Tempo total: {fim - inicio}s)")


# --------------------------------------------------------------------------------------------------


def reconhecimento_faces(localizacao_imagem):
    """
    Realiza o processo de reconhecimento de face na imagem especificada.
    """

    global dataset      # Obtem a referencia da variavel dataset

    # Obtem a imagem a partir do diretorio
    imagem = face_recognition.load_image_file(localizacao_imagem)

    # Obtem as coordenadas do bounding box da face localizada
    localizacao_face = face_recognition.face_locations(imagem)

    # Obtem a codificacao da face localizada
    codificacao_face = face_recognition.face_encodings(imagem, localizacao_face)

    # Auxilia no desenho do bounding box
    imagem_pillow = Image.fromarray(imagem)
    desenho = ImageDraw.Draw(imagem_pillow)

    # Realiza o reconhecimento da face de acordo com o dataset e apresenta a imagem com o bounding
    # box desenhado na face reconhecida além do nome
    for bounding_box, codificacao_desconhecida in zip(localizacao_face, codificacao_face):
        nome = compara_face(codificacao_desconhecida)
        if not nome:
            nome = "Desconhecido"
        apresenta_face(desenho, bounding_box, nome)

    del desenho
    imagem_pillow.show()


# --------------------------------------------------------------------------------------------------


def compara_face(codificacao_desconhecida):
    """
    Compara a codificacao da face detectada na nova imagem, com as codificacoes do dataset e retorna
    o nome da classificacao de maior compatibilidade.
    """

    global dataset   # Obtem a referencia da variavel dataset

    # Obtem uma lista de booleanos correspondentes a verificacao da nova codificacao no dataset
    compatibilidade = face_recognition.compare_faces(dataset["codificacoes"], codificacao_desconhecida)

    # Obtem o nome da classificacao com maior compatibilidade
    votos = Counter(nome for match, nome in zip(compatibilidade, dataset["nomes"]) if match)
    if votos:
        return votos.most_common(1)[0][0]


# --------------------------------------------------------------------------------------------------
    

def apresenta_face(desenho, bounding_box, nome):
    """
    Apresenta a imagem com o bounding box e a legenda desenhados na face detectada e classificada.
    """
    
    # Obtem as coordenadas para desenho do bounding box
    top, right, bottom, left = bounding_box
    # Desenha o bounding box da face
    desenho.rectangle(((left, top), (right, bottom)), outline=COR_BOUNDING_BOX)
    
    # Obtem as coordenadas para escrita do texto
    text_left, text_top, text_right, text_bottom = desenho.textbbox((left, bottom), nome)
    # Desenha o retangulo preenchido onde ficará o texto, abaixo do bounding box
    desenho.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=COR_BOUNDING_BOX,
                      outline=COR_BOUNDING_BOX)
    # Desenha o texto dentro do retangulo
    desenho.text((text_left, text_top), nome, fill=COR_TEXTO)


# --------------------------------------------------------------------------------------------------


def validacao(diretorio = "validacao"):
    """
    Realiza a validacao do modelo a partir de imagens de teste.
    """

    print("\n[VALIDACAO] Iniciando validacao de modelo...")

    inicio = time.time()

    # Percorre todas as imagens do diretorio especificado, realizando o reconhecimento de faces
    for imagem_validacao in Path(diretorio).rglob("*"):
        if imagem_validacao.is_file():
            reconhecimento_faces(str(imagem_validacao.absolute()))

    fim = time.time()

    print(f"\n[VALIDACAO] Validacao de modelo finalizada. (Tempo total: {fim - inicio}s)")


# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    print("\n[DETECTOR] Iniciando programa...")

    inicio = time.time()

    treinamento()
    validacao("input")

    fim = time.time()

    print(f"\n[DETECTOR] Programa finalizado. (Tempo total de execucao: {fim - inicio}s)\n\n")

    