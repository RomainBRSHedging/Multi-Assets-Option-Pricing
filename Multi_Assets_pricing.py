import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

RNG = np.random.default_rng()

# Market Data
stocks = ['AAPL', 'AMZN', 'SPY', 'MSFT']
df = yf.download(tickers=stocks, start= '2022-11-29' , end='2023-11-30')
Stocks_Returns = df.Close.pct_change(1).dropna()
# Correlation Matrix
Corr_Matrix = Stocks_Returns.corr() # Symetrix Positive Definite
Corr_Matrix = np.array(Corr_Matrix)
# print( 'Correlation Matrix: ')
# print(Corr_Matrix)

# Cholesky Decomposition of Corr_Matrix
L = np.linalg.cholesky(Corr_Matrix) # Tri inf
L_t = L.T 
        # Test
# print(np.dot(L, L_t) == Corr_Matrix) 
# print(np.dot(L, L_t))

NB_stocks = len(stocks)                                  # Number of correlated stocks 
T = 252                                                  # Number of Days
N = 1                                                    # Number of Years
M = 1000                                                 # Number of Paths
sigma = 0.02                                             # Volatility 
r = 0.001                                                # Risk-free rate 




def Prix_MC(T,N,M,sigma,r):

    dt = N/T                                                 # Time increment
    S0_m = np.array(list(df['Close'].iloc[0]))               # Initial price of each stocks
    St = np.full((NB_stocks, T, M), 100.0)                   # Stocks prices initialization
    for i in range(NB_stocks):
        St[i, :, :] = S0_m[i]


    for t in range(1, T):
        # Random standard normal
        random_array = RNG.normal(0, 1, (NB_stocks, M))
        
        # To obtain correlated epsilons
        epsilon_corr = np.dot(L, random_array)

        # Sample price path per stock
        for n in range(NB_stocks):
            S = St[n, t - 1, :]
            W_corr = epsilon_corr[n, :]
            
            # Generate new stock price
            St[n, t, :] = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W_corr)



    # Payoff Best of
    ST = St[:,-1,:]
    payoff = np.ones((NB_stocks, M))
    for i in range(NB_stocks):  
        payoff[i,:] = np.where(ST[i,:]-S0_m[i]>0, ST[i,:]-S0_m[i], 0)

    mean_payoff = np.mean(payoff, axis=1)
    std_payoff = np.mean(np.std(payoff, axis=1))
    Prix_Best_of = max(mean_payoff)*np.exp(-r*T)


    # Plot simulated price paths
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # Exemple de traçage en 3D avec ajout de label
    # sc = ax.scatter3D(x, y, z, label='Points 3D')


    # Trajectoires pour chaque action
    for n in range(NB_stocks):
        ax.plot(range(T), St[n, :])

    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Paths')
    ax.legend()

    plt.show()

    return Prix_Best_of, std_payoff




if __name__ == '__main__':

    Prix_Best_0f_Call, std_BO_Call = Prix_MC(T,N,M,sigma,r)
    print('Prix Best Of Call: ', Prix_Best_0f_Call)

    # Calcul de la moyenne et de l'écart type
    ecart_type = std_BO_Call#, ddof=1)  # Utilisez ddof=1 pour calculer l'écart type de l'échantillon

    # Choisissez le niveau de confiance (par exemple, 95%)
    niveau_confiance = 0.95
    coefficient = 1.96  # Pour un niveau de confiance de 95%

    # Calcul de l'intervalle de confiance
    if M > 1:
        intervalle_confiance = (Prix_Best_0f_Call - coefficient * ecart_type / np.sqrt(M),
                                Prix_Best_0f_Call + coefficient * ecart_type / np.sqrt(M))
        print("Intervalle de Confiance 95% :", '[',intervalle_confiance[0],intervalle_confiance[1], ']' )
    else:
        print("L'échantillon est trop petit pour calculer l'intervalle de confiance.")



  



