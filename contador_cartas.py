import cv2
import numpy as np
import os
import re

# Configurações de dimensões para normalizar a carta
LARGURA_CARTA = 200
ALTURA_CARTA = 300

# Dimensões da área do canto (índice: número + naipe) da Copag 157
CANTO_LARGURA = 35
CANTO_ALTURA = 85

PASTA_REFERENCIAS = "referencias"
if not os.path.exists(PASTA_REFERENCIAS):
    os.makedirs(PASTA_REFERENCIAS)

def ordenar_pontos(pts):
    """Ordena os 4 cantos da carta: topo-esq, topo-dir, baixo-dir, baixo-esq."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def achatar_carta(imagem, pontos):
    """Corrige a perspectiva da carta inclinada para uma visão perfeitamente reta."""
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

def verificar_virada_para_baixo(carta_reta):
    """
    Verifica se a carta está virada para baixo analisando o brilho médio do centro.
    A frente da carta Copag é branca (brilho alto); o verso vermelho/azul é escuro.
    """
    centro = carta_reta[60:240, 40:160]
    cinza = cv2.cvtColor(centro, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(cinza)
    # Se o brilho médio for menor que 135, é o verso colorido da carta
    return brilho_medio < 135

def processar_canto(carta_reta):
    """Extrai e binariza (preto e branco puro) o canto superior esquerdo."""
    canto = carta_reta[5:CANTO_ALTURA, 5:CANTO_LARGURA]
    cinza = cv2.cvtColor(canto, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarizada

def carregar_referencias():
    """Carrega múltiplas referências agrupadas pelo nome limpo da carta."""
    refs = {}
    total_arquivos = 0
    for arquivo in os.listdir(PASTA_REFERENCIAS):
        if arquivo.lower().endswith(".png"):
            nome_sem_ext = os.path.splitext(arquivo)[0]
            nome_carta = re.sub(r'[_\-\s]*\d+$', '', nome_sem_ext)
            if not nome_carta:
                nome_carta = nome_sem_ext

            img = cv2.imread(os.path.join(PASTA_REFERENCIAS, arquivo), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if nome_carta not in refs:
                    refs[nome_carta] = []
                refs[nome_carta].append(img)
                total_arquivos += 1
    return refs, total_arquivos

def salvar_referencia(nome_carta, imagem_canto):
    """Salva uma nova variação da carta sem sobrescrever as existentes."""
    nome_limpo = re.sub(r'[_\-\s]*\d+$', '', nome_carta) or nome_carta
    contador = 1
    while True:
        caminho = os.path.join(PASTA_REFERENCIAS, f"{nome_limpo}-{contador}.png")
        if not os.path.exists(caminho):
            cv2.imwrite(caminho, imagem_canto)
            return caminho
        contador += 1

def extrair_valor_carta(nome_carta):
    """
    Extrai o símbolo principal da carta (ex: 'A_Espadas' -> 'A', '10_Copas' -> '10')
    e retorna seu valor no Blackjack e seu peso na contagem Hi-Lo.
    """
    if nome_carta in ["Virada_Baixo", "Desconhecida", "Sem Referencia"]:
        return "0", 0, 0

    simbolo = nome_carta.split("_")[0].upper()

    # Valor base no Blackjack (Ás começa valendo 11)
    if simbolo == "A":
        valor_bj = 11
    elif simbolo in ["K", "Q", "J", "10"]:
        valor_bj = 10
    elif simbolo.isdigit():
        valor_bj = int(simbolo)
    else:
        return simbolo, 0, 0

    # Contagem Hi-Lo (Valor do Deck)
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
    """
    Calcula o total da mão do jogador aplicando a regra dinâmica do Ás (11 ou 1).
    """
    total = 0
    ases = 0

    for nome in lista_cartas:
        simbolo, valor_bj, _ = extrair_valor_carta(nome)
        total += valor_bj
        if simbolo == "A":
            ases += 1

    # Se estourar 21 e houver Ases valendo 11, converte cada Ás para 1 (subtrai 10)
    while total > 21 and ases > 0:
        total -= 10
        ases -= 1

    return total

def identificar_carta(carta_reta, referencias):
    """Verifica se está virada para baixo ou identifica o índice nos dois sentidos."""
    if verificar_virada_para_baixo(carta_reta):
        return "Virada_Baixo", 0

    if not referencias:
        return "Sem Referencia", 0

    canto_normal = processar_canto(carta_reta)
    carta_180 = cv2.rotate(carta_reta, cv2.ROTATE_180)
    canto_invertido = processar_canto(carta_180)
    
    candidatos = [canto_normal, canto_invertido]

    melhor_nome = "Desconhecida"
    menor_diferenca = float("inf")
    LIMIAR_CONFIANCA = 850 

    for canto_atual in candidatos:
        for nome, lista_imgs in referencias.items():
            for img_ref in lista_imgs:
                if canto_atual.shape != img_ref.shape:
                    continue
                    
                diff = cv2.absdiff(canto_atual, img_ref)
                pontuacao = np.sum(diff) / 255
                
                if pontuacao < menor_diferenca:
                    menor_diferenca = pontuacao
                    melhor_nome = nome

    if menor_diferenca == float("inf"):
        return "Desconhecida", 0

    if menor_diferenca < LIMIAR_CONFIANCA:
        return melhor_nome, int(menor_diferenca)
    return "Desconhecida", int(menor_diferenca)

def desenhar_painel_hud(frame, valor_mao, valor_deck_acumulado, cartas_na_mesa):
    """Desenha um painel limpo no canto superior esquerdo da tela com os contadores."""
    # Fundo semitransparente escuro para facilitar a leitura
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (310, 135), (0, 255, 0), 2)

    # Cor dinâmica para o valor da mão (Vermelho se estourar 21, Dourado se for 21)
    if valor_mao > 21:
        cor_mao = (0, 0, 255)      # Vermelho (Estourou / Bust)
        status_mao = f"{valor_mao} (ESTOUROU)"
    elif valor_mao == 21:
        cor_mao = (0, 215, 255)    # Dourado (21!)
        status_mao = "21!"
    else:
        cor_mao = (255, 255, 255)  # Branco
        status_mao = str(valor_mao)

    # Cor dinâmica para o Valor do Deck (Verde se positivo, Vermelho se negativo)
    if valor_deck_acumulado > 0:
        cor_deck = (0, 255, 0)
        texto_deck = f"+{valor_deck_acumulado}"
    elif valor_deck_acumulado < 0:
        cor_deck = (0, 80, 255)
        texto_deck = str(valor_deck_acumulado)
    else:
        cor_deck = (200, 200, 200)
        texto_deck = "0"

    cv2.putText(frame, f"Mao Jogador: {status_mao}", (22, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor_mao, 2)
    cv2.putText(frame, f"Valor do Deck: {texto_deck}", (22, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor_deck, 2)
    cv2.putText(frame, f"Cartas na mesa: {len(cartas_na_mesa)}", (22, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

def main():
    cap = cv2.VideoCapture(0)
    referencias, total_fotos = carregar_referencias()
    
    print("=== CONTADOR DE BLACKJACK COPAG 157 ===")
    print(f"{len(referencias)} cartas ({total_fotos} fotos) carregadas.")
    print("Controles na janela do video:")
    print("  [C] - Cadastrar nova carta/variacao")
    print("  [ESPACO] - Registrar cartas atuais da mesa no Valor do Deck (fim da rodada)")
    print("  [R] - Resetar o Valor do Deck para 0 (baralho embaralhado)")
    print("  [Q] - Sair")

    ultima_carta_registrada = ""
    ultimo_canto_valido = None
    
    valor_deck_acumulado = 0
    cartas_ja_contadas_na_rodada = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cartas_detectadas_frame = []

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 5000:
                perimetro = cv2.arcLength(cnt, True)
                aprox = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)

                if len(aprox) == 4:
                    pontos = aprox.reshape(4, 2)
                    carta_reta = achatar_carta(frame, pontos)
                    nome_carta, erro = identificar_carta(carta_reta, referencias)

                    if nome_carta != "Virada_Baixo":
                        ultimo_canto_valido = processar_canto(carta_reta)
                        cv2.imshow("Canto Processado", cv2.resize(ultimo_canto_valido, (105, 255)))

                    cartas_detectadas_frame.append(nome_carta)

                    # Define cores e textos por status
                    if nome_carta == "Virada_Baixo":
                        cor = (255, 150, 0) # Azul para carta virada
                        texto = "Virada (0)"
                    elif nome_carta in ["Desconhecida", "Sem Referencia"]:
                        cor = (0, 255, 255) # Amarelo
                        texto = f"{nome_carta}"
                    else:
                        cor = (0, 255, 0)   # Verde
                        _, val_bj, peso_hl = extrair_valor_carta(nome_carta)
                        sinal = "+" if peso_hl > 0 else ""
                        texto = f"{nome_carta} [BJ:{val_bj} | Deck:{sinal}{peso_hl}]"

                    cv2.drawContours(frame, [aprox], -1, cor, 3)
                    x, y = pontos[0]
                    cv2.putText(frame, texto, (int(x), int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

        # Calcula a mão atual em tempo real
        valor_mao_atual = calcular_mao_blackjack(cartas_detectadas_frame)
        
        # Soma temporária das cartas que estão visíveis agora na mesa ao deck acumulado
        delta_mesa_atual = sum(extrair_valor_carta(c)[2] for c in cartas_detectadas_frame)
        valor_deck_exibido = valor_deck_acumulado + delta_mesa_atual

        # Desenha o painel no canto superior esquerdo
        desenhar_painel_hud(frame, valor_mao_atual, valor_deck_exibido, cartas_detectadas_frame)

        cv2.imshow("Detector Copag 157 - Blackjack", frame)
        tecla = cv2.waitKey(1) & 0xFF

        # [ESPACO] Consolida as cartas da mesa atual no histórico do deck antes de recolhê-las
        if tecla == ord(' '):
            valor_deck_acumulado += delta_mesa_atual
            print(f"\n[RODADA REGISTRADA] Valor do Deck consolidado em: {valor_deck_acumulado}")

        # [R] Reseta o contador do deck (quando o dealer embaralhar)
        elif tecla in [ord('r'), ord('R')]:
            valor_deck_acumulado = 0
            print("\n[RESET] Valor do Deck zerado!")

        # [C] Cadastra nova carta
        elif tecla in [ord('c'), ord('C')]:
            if ultimo_canto_valido is None:
                print("\n[AVISO] Nenhuma carta aberta detectada na tela.")
            else:
                sugestao = f" [Enter para repetir '{ultima_carta_registrada}']" if ultima_carta_registrada else ""
                entrada = input(f"\nNome da carta (ex: A_Espadas, K_Copas, 5_Ouros){sugestao}: ").strip()
                nome_final = entrada if entrada else ultima_carta_registrada

                if nome_final:
                    caminho_salvo = salvar_referencia(nome_final, ultimo_canto_valido)
                    ultima_carta_registrada = nome_final
                    referencias, total_fotos = carregar_referencias()
                    nome_limpo = re.sub(r'[_\-\s]*\d+$', '', nome_final) or nome_final
                    print(f"Salvo! Total de fotos para {nome_limpo}: {len(referencias[nome_limpo])}")
                    print(">> CLIQUE NA JANELA DO VIDEO PARA CONTINUAR <<")

        elif tecla in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()