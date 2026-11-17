import cv2
import numpy as np
import os
import re

# Tamanho padrão para deixar todas as cartas retas na mesma escala
LARGURA_CARTA = 200
ALTURA_CARTA = 300

# Tamanho da área do canto superior esquerdo (onde fica o número e o naipe)
CANTO_LARGURA = 35
CANTO_ALTURA = 85

# Cria a pasta de referências caso ela ainda não exista
PASTA_REFERENCIAS = "referencias"
if not os.path.exists(PASTA_REFERENCIAS):
    os.makedirs(PASTA_REFERENCIAS)

def ordenar_pontos(pts):
    """Organiza os cantos da carta na ordem: topo-esquerdo, topo-direito, baixo-direito e baixo-esquerdo."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def achatar_carta(imagem, pontos):
    """Estica e endireita a carta inclinada para que ela fique vista perfeitamente de cima."""
    rect = ordenar_pontos(pontos)
    dst = np.array([
        [0, 0],
        [LARGURA_CARTA - 1, 0],
        [LARGURA_CARTA - 1, ALTURA_CARTA - 1],
        [0, ALTURA_CARTA - 1]
    ], dtype="float32")
    
    matriz = cv2.getPerspectiveTransform(rect, dst)
    carta_reta = cv2.warpPerspective(imagem, matriz, (LARGURA_CARTA, ALTURA_CARTA))
    return carta_reta

def detectar_cor_vermelha(carta_reta):
    """Verifica se a cor do número/naipe é vermelha (Ouros/Copas) ou preta (Paus/Espadas)."""
    canto_bgr = carta_reta[5:CANTO_ALTURA, 5:CANTO_LARGURA]
    hsv = cv2.cvtColor(canto_bgr, cv2.COLOR_BGR2HSV)
    
    # Filtra as tonalidades de vermelho na imagem
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return cv2.countNonZero(mask1 | mask2) > 35

def processar_canto(carta_reta):
    """Recorta o canto da carta e transforma em preto e branco puro para facilitar a leitura."""
    canto = carta_reta[5:CANTO_ALTURA, 5:CANTO_LARGURA]
    cinza = cv2.cvtColor(canto, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarizada

def centralizar_simbolo(img_bin):
    """Recorta o desenho branco e o coloca bem no meio de um quadrado padrão."""
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return np.zeros((36, 36), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    recorte = img_bin[y:y+h, x:x+w]
    
    # Ajusta o tamanho do símbolo sem esticar ou deformar
    escala = min(30.0 / max(w, 1), 30.0 / max(h, 1))
    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))
    redim = cv2.resize(recorte, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    # Cola o símbolo centralizado no fundo preto
    tela = np.zeros((36, 36), dtype=np.uint8)
    off_x = (36 - novo_w) // 2
    off_y = (36 - novo_h) // 2
    tela[off_y:off_y+novo_h, off_x:off_x+novo_w] = redim
    return tela

def extrair_rank_e_naipe(canto_bin):
    """Separa o canto em duas partes: o número/letra em cima e o naipe embaixo."""
    rank_raw = canto_bin[0:42, :]
    naipe_raw = canto_bin[40:80, :]
    return centralizar_simbolo(rank_raw), centralizar_simbolo(naipe_raw)

def carregar_referencias():
    """Lê todas as imagens salvas na pasta e organiza cada carta pelo nome limpo."""
    refs = {}
    total_arquivos = 0
    for arquivo in os.listdir(PASTA_REFERENCIAS):
        if arquivo.lower().endswith(".png"):
            nome_sem_ext = os.path.splitext(arquivo)[0]
            
            # Limpa sufixos e números do nome do arquivo para agrupar variações da mesma carta
            nome_carta = re.sub(r'(_aug)?[_\-\s]*\d+$', '', nome_sem_ext)
            if not nome_carta:
                nome_carta = nome_sem_ext

            img = cv2.imread(os.path.join(PASTA_REFERENCIAS, arquivo), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                rank_ref, naipe_ref = extrair_rank_e_naipe(img_bin)
                if nome_carta not in refs:
                    refs[nome_carta] = []
                refs[nome_carta].append((rank_ref, naipe_ref))
                total_arquivos += 1
    return refs, total_arquivos

def salvar_referencia(nome_carta, imagem_canto):
    """Salva a foto do canto na pasta sem apagar as imagens que já existem."""
    nome_limpo = re.sub(r'(_aug)?[_\-\s]*\d+$', '', nome_carta) or nome_carta
    
    contador = 1
    while True:
        caminho = os.path.join(PASTA_REFERENCIAS, f"{nome_limpo}-{contador}.png")
        if not os.path.exists(caminho):
            cv2.imwrite(caminho, imagem_canto)
            return caminho
        contador += 1 

def identificar_carta(carta_reta, referencias):
    """Compara a carta da câmera (normal e de cabeça para baixo) com as referências salvas."""
    if not referencias:
        return "Sem Referencia", 0

    # Prepara a imagem normal e rotacionada para ler a carta em qualquer posição
    carta_180 = cv2.rotate(carta_reta, cv2.ROTATE_180)
    candidatos = [carta_reta, carta_180]

    melhor_nome = "Desconhecida"
    melhor_score = -1.0

    for img_candidata in candidatos:
        eh_vermelha = detectar_cor_vermelha(img_candidata)
        canto_bin = processar_canto(img_candidata)
        rank_atual, naipe_atual = extrair_rank_e_naipe(canto_bin)

        for nome, lista_pares in referencias.items():
            nome_lower = nome.lower()
            ref_vermelha = ("ouros" in nome_lower) or ("copas" in nome_lower)
            ref_preta = ("paus" in nome_lower) or ("espadas" in nome_lower)

            # Pula a comparação se a cor da carta não bater com a cor da referência
            if eh_vermelha and ref_preta:
                continue
            if not eh_vermelha and ref_vermelha:
                continue

            # Calcula o quanto o número e o naipe se parecem com cada referência
            for rank_ref, naipe_ref in lista_pares:
                score_rank = cv2.matchTemplate(rank_atual, rank_ref, cv2.TM_CCOEFF_NORMED)[0][0]
                score_naipe = cv2.matchTemplate(naipe_atual, naipe_ref, cv2.TM_CCOEFF_NORMED)[0][0]

                score_final = (score_rank * 0.5) + (score_naipe * 0.5)

                if score_final > melhor_score:
                    melhor_score = score_final
                    melhor_nome = nome

    # Transforma a semelhança em uma pontuação de erro para mostrar na tela
    erro_calculado = int((1.0 - max(0.0, melhor_score)) * 1000)

    # Só confirma a carta se a semelhança mínima for atingida
    if melhor_score >= 0.50:
        return melhor_nome, erro_calculado
    return "Desconhecida", erro_calculado

def main():
    # Liga a webcam e carrega as imagens de referência
    cap = cv2.VideoCapture(0)
    referencias, total_fotos = carregar_referencias()
    print(f"Sistema iniciado. {len(referencias)} cartas ({total_fotos} fotos de referencia) carregadas.")
    print("Pressione 'C' para cadastrar uma variação da carta | 'Q' para sair.")

    ultima_carta_registrada = ""
    ultimo_canto_valido = None

    while True:
        # Lê cada quadro do vídeo da câmera
        ret, frame = cap.read()
        if not ret:
            break

        # Prepara a imagem para encontrar o contorno branco da carta na mesa
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            # Ignora objetos pequenos e foca apenas no tamanho de uma carta
            if area > 5000:
                perimetro = cv2.arcLength(cnt, True)
                aprox = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)

                # Confirma se o objeto encontrado tem formato retangular (4 cantos)
                if len(aprox) == 4:
                    pontos = aprox.reshape(4, 2)
                    carta_reta = achatar_carta(frame, pontos)
                    ultimo_canto_valido = processar_canto(carta_reta)

                    nome_carta, erro = identificar_carta(carta_reta, referencias)

                    # Pinta a borda de verde se reconheceu a carta, ou amarelo se ainda não sabe qual é
                    cor = (0, 255, 0) if nome_carta not in ["Desconhecida", "Sem Referencia"] else (0, 255, 255)
                    
                    cv2.drawContours(frame, [aprox], -1, cor, 3)
                    x, y = pontos[0]
                    texto = f"{nome_carta} (Erro: {erro})"
                    cv2.putText(frame, texto, (int(x), int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                    
                    # Mostra a janela pequena com o recorte preto e branco do canto
                    cv2.imshow("Canto Processado", cv2.resize(ultimo_canto_valido, (105, 255)))

        # Exibe a imagem principal da câmera
        cv2.imshow("Detector Copag 157", frame)
        tecla = cv2.waitKey(1) & 0xFF

        # Salva uma nova foto de referência ao apertar a tecla C
        if tecla in [ord('c'), ord('C')] and ultimo_canto_valido is not None:
            sugestao = f" [Enter para repetir '{ultima_carta_registrada}']" if ultima_carta_registrada else ""
            entrada = input(f"\nNome da carta (ex: A_Espadas){sugestao}: ").strip()
            
            nome_final = entrada if entrada else ultima_carta_registrada

            if nome_final:
                caminho_salvo = salvar_referencia(nome_final, ultimo_canto_valido)
                ultima_carta_registrada = nome_final
                referencias, total_fotos = carregar_referencias()
                nome_limpo = re.sub(r'(_aug)?[_\-\s]*\d+$', '', nome_final) or nome_final
                qtd_var = len(referencias[nome_limpo])
                print(f"Salvo em '{caminho_salvo}'! Total de variações para {nome_limpo}: {qtd_var}")

        # Encerra o programa ao apertar a tecla Q
        elif tecla in [ord('q'), ord('Q')]:
            break

    # Desliga a câmera e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()