import numpy as np
import matplotlib.pyplot as plt
import time

# macierz przejscia
t1 = ['P', 'K', 'N']
# macierz wystapien
wyst = np.array([[2, 4, 0], [0, 0, 4], [4, 0, 2]])
# strategia przeciwnika
prob_op = np.array([0.5, 0.1, 0.4])

n = 30
state = 'P'
wynik = 0
stan_kasy = np.array([0])

def turn(idx):
    global state, wynik, stan_kasy
    p_wyst = wyst[idx] / sum(wyst[idx])
    pred = np.random.choice(t1, p=p_wyst)

    # odpowiedz na predykcje
    op_akc = np.random.choice(t1, p=prob_op)

    # zobacz rzeczywista akcje przeciwnika
    # zobaczyć wynik gry - pred, op_akc
    print("Predykcja: " + pred)
    print("Macierz prawdopodobieństwa wystąpień: ")
    print(p_wyst)
    print("Rzeczywisty wybór: " + op_akc)
    if pred == op_akc:
        print("Wygrana")
        wynik += 1
    elif pred == "P" and op_akc == "K":
        print("Przegrana")
        wynik -= 1
    elif pred == "K" and op_akc == "N":
        print("Przegrana")
        wynik -= 1
    elif pred == "N" and op_akc == "P":
        print("Przegrana")
        wynik -= 1
    else:
        print("Remis")
    stan_kasy = np.append(stan_kasy, wynik)

    # zmodyfikować macierz wystapien
    idx2 = 0
    if op_akc == 'P':
        pass
    elif op_akc == 'K':
        idx2 = 1
    else:
        idx2 = 2
    wyst[idx, idx2] += 1

    # przejść do stanu - jaki zagrał przeciwnik: op_akc
    state = op_akc
    print(f"Wynik: {wynik}")
    print("----------\n")


for i in range(n):
    if state == 'P':
        turn(0)

    elif state == 'K':
        turn(1)

    elif state == 'N':
        turn(2)

    # time.sleep(0.5)

plt.plot(stan_kasy)
# ustawianie osi x by skok był w liczbach całkowitych z uwzg rund
plt.xticks(np.arange(0, n+1, 1))
# ustawianie osi y by skok był w liczbach całkowitych z uwzg min i max wyniku
plt.yticks(np.arange(min(stan_kasy), max(stan_kasy)+1, 1))
plt.xlabel('Runda')
plt.ylabel('Stan kasy')
plt.title('Stan kasy na przestrzeni rozgrywki')
plt.grid(True)
plt.show()

