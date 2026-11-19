DETECTOR DE CARTAS COPAG 157 & CONTADOR DE BLACKJACK 21

Sistema em Python com OpenCV para identificar cartas do baralho Copag 157 via webcam e contar pontos de Blackjack (21) e contagem de deck (Hi-Lo) em tempo real.

COMO INSTALAR

1. Instale o Python 3 no computador.
2. Abra o terminal na pasta do projeto e instale as dependencias:
   pip install -r requirements.txt


COMO USAR

1. Cadastro de referencias (opcional, caso precise refazer alguma carta):
   python detector_copag.py
   - Aperte 'C' para salvar a carta atual.
   - Aperte 'Q' para sair.

2. Gerar variacoes das referencias:
   python gerador_variedade.py

3. Iniciar a mesa de Blackjack 21:
   python contador_partida.py
   - Coloque as cartas do Dealer na metade de cima da tela e as do Jogador na metade de baixo.
   - Aperte 'ESPACO' ao fim da rodada para somar as cartas na contagem do Deck.
   - Aperte 'R' para zerar a contagem do Deck quando embaralhar.
   - Aperte 'Q' para sair.
