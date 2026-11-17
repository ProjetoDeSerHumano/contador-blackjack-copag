import cv2
import numpy as np
import os
import re

# Medidas padrão usadas para deixar todas as cartas retas na mesma escala
LARGURA_CARTA = 200
ALTURA_CARTA = 300
CANTO_LARGURA = 35
CANTO_ALTURA = 85

# Pasta onde ficam guardadas as imagens de referência das cartas
PASTA_REFERENCIAS = "referencias"

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
    """Estica e endireita a carta inclinada na mesa para uma visão perfeitamente reta de cima."""
    rect = ordenar_pontos(pontos)
    dst = np.array([
        [0, 0],
        [LARGURA_CARTA - 1, 0],
        [LARGURA_CARTA - 1, ALTURA_CARTA - 1],
        [0, ALTURA_CARTA - 1]
    ], dtype="float32")
    matriz = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(imagem, matriz, (LARGURA_CARTA, ALTURA_CARTA))

def verificar_virada_para_baixo(carta_reta):
    """Analisa se o centro da carta é escuro (verso colorido) ou claro (frente branca)."""
    centro = carta_reta[60:240, 40:160]
    cinza = cv2.cvtColor(centro, cv2.COLOR_BGR2GRAY)
    return np.mean(cinza) < 135

def detectar_cor_vermelha(carta_reta):
    """Identifica se o símbolo no canto é vermelho (Ouros/Copas) ou preto (Paus/Espadas)."""
    canto_bgr = carta_reta[5:CANTO_ALTURA, 5:CANTO_LARGURA]
    hsv = cv2.cvtColor(canto_bgr, cv2.COLOR_BGR2HSV)
    
    # Filtra as faixas de cor vermelha para separar os naipes por cor
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return cv2.countNonZero(mask1 | mask2) > 35

def processar_canto(carta_reta):
    """Recorta o canto superior esquerdo e transforma em preto e branco puro."""
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
    
    # Redimensiona mantendo a proporção original sem esticar ou deformar o desenho
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
    """Divide o recorte do canto em duas partes: número/letra em cima e naipe embaixo."""
    rank_raw = canto_bin[0:42, :]
    naipe_raw = canto_bin[40:80, :]
    return centralizar_simbolo(rank_raw), centralizar_simbolo(naipe_raw)

def carregar_referencias():
    """Lê todas as imagens da pasta de referências e prepara os desenhos de número e naipe."""
    refs = {}
    if not os.path.exists(PASTA_REFERENCIAS):
        return refs

    for arquivo in os.listdir(PASTA_REFERENCIAS):
        if arquivo.lower().endswith(".png"):
            nome_sem_ext = os.path.splitext(arquivo)[0]
            nome_carta = re.sub(r'[_\-\s]*\d+$', '', nome_sem_ext) or nome_sem_ext

            img = cv2.imread(os.path.join(PASTA_REFERENCIAS, arquivo), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                rank_ref, naipe_ref = extrair_rank_e_naipe(img_bin)
                if nome_carta not in refs:
                    refs[nome_carta] = []
                refs[nome_carta].append((rank_ref, naipe_ref))
    return refs

def identificar_carta(carta_reta, referencias):
    """Descobre qual é a carta comparando cor, número e naipe nos dois sentidos de leitura."""
    if verificar_virada_para_baixo(carta_reta):
        return "Virada_Baixo"
    if not referencias:
        return "Sem Referencia"

    # Testa a carta na posição normal e girada de cabeça para baixo
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

            # Ignora referências cuja cor não combina com a cor detectada na câmera
            if eh_vermelha and ref_preta:
                continue
            if not eh_vermelha and ref_vermelha:
                continue

            # Mede a semelhança do número e do naipe separadamente
            for rank_ref, naipe_ref in lista_pares:
                score_rank = cv2.matchTemplate(rank_atual, rank_ref, cv2.TM_CCOEFF_NORMED)[0][0]
                score_naipe = cv2.matchTemplate(naipe_atual, naipe_ref, cv2.TM_CCOEFF_NORMED)[0][0]

                score_final = (score_rank * 0.5) + (score_naipe * 0.5)

                if score_final > melhor_score:
                    melhor_score = score_final
                    melhor_nome = nome

    # Só confirma o nome da carta se atingir o nível mínimo de confiança
    if melhor_score >= 0.50:
        return melhor_nome
    return "Desconhecida"

def extrair_valor_carta(nome_carta):
    """Converte o nome da carta em seu valor de pontos no Blackjack e seu peso na contagem do baralho."""
    if nome_carta in ["Virada_Baixo", "Desconhecida", "Sem Referencia"]:
        return "0", 0, 0

    simbolo = nome_carta.split("_")[0].upper()

    # Define quanto a carta vale na soma da mão
    if simbolo == "A":
        valor_bj = 11
    elif simbolo in ["K", "Q", "J", "10"]:
        valor_bj = 10
    elif simbolo.isdigit():
        valor_bj = int(simbolo)
    else:
        return simbolo, 0, 0

    # Define o peso da carta na contagem Hi-Lo (cartas baixas somam, altas subtraem)
    if simbolo in ["2", "3", "4", "5", "6"]:
        peso_hilo = 1
    elif simbolo in ["7", "8", "9"]:
        peso_hilo = 0
    elif simbolo in ["10", "J", "Q", "K", "A"]:
        peso_hilo = -1
    else:
        peso_hilo = 0

    return simbolo, valor_bj, peso_hilo

def calcular_mao_blackjack(lista_cartas):
    """Soma os pontos das cartas e ajusta o Ás automaticamente se passar de 21."""
    total = 0
    ases = 0
    for nome in lista_cartas:
        simbolo, valor_bj, _ = extrair_valor_carta(nome)
        total += valor_bj
        if simbolo == "A":
            ases += 1

    # Converte Ás de 11 para 1 sempre que a soma ultrapassar 21
    while total > 21 and ases > 0:
        total -= 10
        ases -= 1
    return total

def determinar_status_partida(pts_jogador, pts_dealer, cartas_jogador, cartas_dealer):
    """Compara as mãos do Jogador e do Dealer para mostrar quem está ganhando na rodada."""
    if not cartas_jogador and not cartas_dealer:
        return "Aguardando cartas...", (200, 200, 200)
    if pts_jogador > 21:
        return "JOGADOR ESTOUROU! (Dealer Vence)", (0, 0, 255)
    if "Virada_Baixo" in cartas_dealer:
        return "Em jogo (Carta do Dealer oculta)", (0, 255, 255)
    if pts_dealer > 21:
        return "DEALER ESTOUROU! (Jogador Vence)", (0, 255, 0)

    if cartas_jogador and cartas_dealer:
        if pts_jogador == 21 and len(cartas_jogador) == 2 and not (pts_dealer == 21 and len(cartas_dealer) == 2):
            return "BLACKJACK DO JOGADOR!", (0, 215, 255)
        if pts_jogador > pts_dealer:
            return "JOGADOR VENCENDO", (0, 255, 0)
        elif pts_dealer > pts_jogador:
            return "DEALER VENCENDO", (0, 100, 255)
        else:
            return "EMPATE (PUSH)", (255, 255, 0)
    return "Em andamento...", (255, 255, 255)

def desenhar_interface_mesa(frame, pts_dealer, pts_jogador, valor_deck, cartas_dealer, cartas_jogador):
    """Desenha na tela a linha que divide a mesa e os placares do Dealer, Jogador e Deck."""
    altura, largura, _ = frame.shape
    meio_y = altura // 2

    # Desenha uma linha tracejada horizontal no meio da tela
    for x in range(0, largura, 25):
        cv2.line(frame, (x, meio_y), (x + 12, meio_y), (255, 255, 255), 2)

    # Cria os fundos escuros semitransparentes para facilitar a leitura dos textos
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (largura - 10, 85), (15, 15, 15), -1)
    cv2.rectangle(overlay, (10, altura - 75), (340, altura - 10), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Desenha as bordas coloridas dos painéis
    cv2.rectangle(frame, (10, 10), (largura - 10, 85), (100, 100, 255), 2)
    cv2.rectangle(frame, (10, altura - 75), (340, altura - 10), (0, 255, 100), 2)

    # Mostra a pontuação do Dealer e o status atual da rodada no painel superior
    cor_dealer = (0, 0, 255) if pts_dealer > 21 else (255, 255, 255)
    txt_dealer = f"DEALER: {pts_dealer}" + (" (ESTOUROU)" if pts_dealer > 21 else "")
    cv2.putText(frame, txt_dealer, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_dealer, 2)

    status_txt, cor_status = determinar_status_partida(pts_jogador, pts_dealer, cartas_jogador, cartas_dealer)
    cv2.putText(frame, status_txt, (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_status, 2)

    # Mostra o saldo acumulado da contagem do baralho no canto superior direito
    if valor_deck > 0:
        cor_deck = (0, 255, 0)
        txt_deck = f"DECK: +{valor_deck}"
    elif valor_deck < 0:
        cor_deck = (0, 80, 255)
        txt_deck = f"DECK: {valor_deck}"
    else:
        cor_deck = (200, 200, 200)
        txt_deck = "DECK: 0"
    cv2.putText(frame, txt_deck, (largura - 210, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cor_deck, 2)

    # Mostra a pontuação da mão do Jogador no painel inferior esquerdo
    if pts_jogador > 21:
        cor_jogador = (0, 0, 255)
        txt_jogador = f"JOGADOR: {pts_jogador} (ESTOUROU)"
    elif pts_jogador == 21:
        cor_jogador = (0, 215, 255)
        txt_jogador = "JOGADOR: 21!"
    else:
        cor_jogador = (0, 255, 100)
        txt_jogador = f"JOGADOR: {pts_jogador}"
    cv2.putText(frame, txt_jogador, (25, altura - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cor_jogador, 2)

def main():
    # Liga a webcam em alta definição e carrega as imagens de referência
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    referencias = carregar_referencias()
    valor_deck_acumulado = 0

    print("=== MESA DE BLACKJACK 21 - COPAG 157 ===")
    print("Controles:")
    print("  [ESPACO] - Registrar fim da rodada no Valor do Deck")
    print("  [R]      - Resetar Valor do Deck para 0")
    print("  [Q]      - Sair")

    # Configura a janela do programa para abrir ocupando a tela inteira
    nome_janela = "Detector Copag 157"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Captura cada quadro de vídeo da câmera
        ret, frame = cap.read()
        if not ret:
            break

        altura_frame, _, _ = frame.shape
        linha_divisoria_y = altura_frame // 2

        # Prepara a imagem para localizar os retângulos brancos das cartas na mesa
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cartas_dealer = []
        cartas_jogador = []

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            # Ignora objetos pequenos e analisa apenas contornos do tamanho de uma carta
            if area > 5000:
                perimetro = cv2.arcLength(cnt, True)
                aprox = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)

                # Verifica se o contorno encontrado possui exatamente quatro cantos
                if len(aprox) == 4:
                    pontos = aprox.reshape(4, 2)
                    
                    # Decide se a carta pertence ao Dealer (metade de cima) ou ao Jogador (metade de baixo)
                    centro_y = np.mean(pontos[:, 1])
                    pertence_ao_dealer = centro_y < linha_divisoria_y

                    carta_reta = achatar_carta(frame, pontos)
                    nome_carta = identificar_carta(carta_reta, referencias)

                    if pertence_ao_dealer:
                        cartas_dealer.append(nome_carta)
                    else:
                        cartas_jogador.append(nome_carta)

                    # Define a cor da borda e o texto exibido acima de cada carta
                    if nome_carta == "Virada_Baixo":
                        cor = (255, 150, 0)
                        texto = "Virada"
                    elif nome_carta in ["Desconhecida", "Sem Referencia"]:
                        cor = (0, 255, 255)
                        texto = nome_carta
                    else:
                        cor = (100, 100, 255) if pertence_ao_dealer else (0, 255, 100)
                        texto = nome_carta

                    cv2.drawContours(frame, [aprox], -1, cor, 3)
                    x, y = pontos[0]
                    cv2.putText(frame, texto, (int(x), int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        # Calcula os pontos atuais de cada lado e a contagem acumulada do baralho
        pts_dealer = calcular_mao_blackjack(cartas_dealer)
        pts_jogador = calcular_mao_blackjack(cartas_jogador)

        todas_cartas_mesa = cartas_dealer + cartas_jogador
        delta_mesa_atual = sum(extrair_valor_carta(c)[2] for c in todas_cartas_mesa)
        valor_deck_exibido = valor_deck_acumulado + delta_mesa_atual

        # Atualiza os placares e a linha divisória na tela
        desenhar_interface_mesa(frame, pts_dealer, pts_jogador, valor_deck_exibido, cartas_dealer, cartas_jogador)

        cv2.imshow(nome_janela, frame)
        tecla = cv2.waitKey(1) & 0xFF

        # Salva a contagem das cartas da rodada atual no saldo do baralho ao apertar Espaço
        if tecla == ord(' '):
            valor_deck_acumulado += delta_mesa_atual
            print(f"[RODADA FINALIZADA] Valor do Deck consolidado em: {valor_deck_acumulado}")
        # Zera a contagem do baralho ao apertar R
        elif tecla in [ord('r'), ord('R')]:
            valor_deck_acumulado = 0
            print("[RESET] Valor do Deck zerado!")
        # Encerra o programa ao apertar Q
        elif tecla in [ord('q'), ord('Q')]:
            break

    # Desliga a câmera e fecha a janela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()