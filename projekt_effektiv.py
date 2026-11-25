import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Simuleret Laserkøling i 2D


# Tilføj universelle værdier + atomernes egenskaber: Såsom masse og atomets egenfrekvens mm.
w_A = 2*np.pi*755e12  # THz (vinkelfrekvens for absorbtion)

Gamma = 400e6  # Hz


delta = -2*Gamma  # detuning/forskel mellem laseren og atomets frekvens
w_L = w_A + delta  # Delta (vinkelfrekvensen for laseren, der er justerbar)

c = 3.0*10**8  # m/s
lambd = (2*np.pi*c)/w_L  # (ca. = 400 nm (bølgelængde, der ikke skal bruges)
# (bølgetallet (også = 2*pi/lambd), anvend i foton momentum og Lorentzfordeling)
k = w_L/c
h_bar = (6.62607015*10**-34)/(2*np.pi)  # J*s, planks konstant/2pi = h bar

p = 100*h_bar*k
masse = 6.64e-26  # kg


# Opgavens grad af tilfældighed for molekylernes start betingelserne
rng = np.random.default_rng()

# Dimension af kasse
d = 10

# start betingelser
N = 100
dt = 0.001


# Tilfældig uniform fordelt position indenfor kassens rammer i både x og y
start_positions = np.array(
    [rng.uniform(-d, d, size=N), rng.uniform(-d, d, size=N)])


# lognormalfordelt tilfældighed for start fart, tager middelværdi og spredning som argumenter
fart = rng.normal(130, 0.03, size=N)
# tilfældig valg med lige stor sandsynlig af fordeling for vinkel i intervallet
vinkel = rng.uniform(0, 2*np.pi, size=N)

# lav array med tilfældig størrelse for v, der tager middelværdi og spredning for v i x-aksen og tilsvarende i y-aksen med en tilfældig retning i x og y
start_velocities = fart * np.array([-np.sin(vinkel), np.cos(vinkel)])


class GasCell:
    def __init__(self, N, start_velocities, start_positions):
        self.N = N
        self.velocities = start_velocities
        self.positions = start_positions
        self.accelerations = np.zeros((2, N))
        self.halveringstid = 5*0.001
        self.tid_siden_laser = np.zeros((N))
        self.color = ["blue"]*N
        self.er_exciteret = np.zeros((N))

    def updatePositions(self, dt):
        """
        Opdaterer atomernes position og vender dem når de rammer kanten, (samme fart, men modsat fortegn på vinkel)

        Args:
            self: Henter atomets informationer
            dt: Tidsskridtet

        Returns:
            Atomernes nye position

        Raises:
            Intet
        """

        self.positions = 0.5*self.accelerations*dt**2+self.velocities*dt+self.positions

        for i in range(len(self.positions[0])):
            if self.positions[0][i] >= d or self.positions[0][i] <= -d:
                self.velocities[0][i] = -self.velocities[0][i]

        for i in range(len(self.positions[1])):
            if self.positions[1][i] >= d or self.positions[1][i] <= -d:
                self.velocities[1][i] = -self.velocities[1][i]

    def photonEmission(self, dt, lifetime=2.5e-9):
        """
        Opdaterer a

                Args:
                    param1: This is the first param.
                    param2: This is a second param.

                Returns:
                    This is a description of what is returned.

                Raises:
                    KeyError: Raises an exception. 
                """
        P = rng.uniform(0, 1, size=N)  # tilfældig energifordeling for atomerne

        n = self.halveringstid/dt  # henfald for atomer per tid

        Q = (0.5)**(1/n)  # sandsynligheden for at et atom ikke henfalder

        kan_henfalde = np.ones((N)) * P > Q

        henfalder = kan_henfalde * self.er_exciteret

        # foton emission i tilfældig retning (atom for spark i tilfældig retning) hvis P>Q
        vinkel = rng.uniform(0, 2*np.pi, N)
        self.velocities += ((h_bar*w_A)/(c*masse)) * \
            np.array([-np.sin(vinkel), np.cos(vinkel)])*henfalder
        self.er_exciteret *= (1-henfalder)
        self.color = np.where(self.er_exciteret, "red", "blue")

        return self.color

    def photonInteraction(self, dt, Gamma=Gamma, w_A=w_A, w_L=w_L, k=k):
        """
        Opdaterer a

        Args:
            param1: This is the first param.
            param2: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """

        def lorentz(w_A, w_L, Gamma, k, v):
            return 1/np.pi * (Gamma/2)/((w_L-(w_A+k*v))**2+(Gamma/2)**2)

        intensity = 1e11

        Ppseudo_x_plus = intensity * \
            lorentz(w_A, w_L, Gamma, k, v=self.velocities[0]) * dt

        Ppseudo_y_plus = intensity * \
            lorentz(w_A, w_L, Gamma, k, v=self.velocities[1]) * dt

        Ppseudo_x_minus = intensity * \
            lorentz(w_A, w_L, Gamma, k, v=-self.velocities[0]) * dt

        Ppseudo_y_minus = intensity * \
            lorentz(w_A, w_L, Gamma, k, v=-self.velocities[1]) * dt

        P = rng.uniform(0, 1, N)
        
        exciteres_xplus = np.ones((N)) * P < Ppseudo_x_plus
        exciteres_xminus = np.ones((N)) * P < Ppseudo_x_minus
        exciteres_yplus = np.ones((N)) * P < Ppseudo_y_plus
        exciteres_yminus = np.ones((N)) * P < Ppseudo_y_minus

        self.velocities[0] += (p / masse) * exciteres_xplus
        self.velocities[0] -= (p / masse) * exciteres_xminus
        self.velocities[1] += (p / masse) * exciteres_yplus
        self.velocities[1] -= (p / masse) * exciteres_yminus

        self.er_exciteret *= exciteres_xplus &  exciteres_xminus & exciteres_yplus & exciteres_yminus
        self.color = np.where(self.er_exciteret, "red", "blue")
        return self.color

    def simulate(self, lifetime=0.5, dt = dt):
        """
        Denne funktion kører selve simuleringen.

        Args:
            self: Henter atomets egenskaber
            dt: Tidskridtet
            Gamma:
            w_A:
            w_L:
            intensity:
            lifetime:

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """

        # Gamma, w_A, w_L, k, intensity, dt = 0.01
        self.photonEmission(dt=dt)
        self.photonInteraction(dt=dt)
        self.updatePositions(dt=dt)

    def getPositions(self):
        """
        Returnerer atomernes positioner

        Args:
            param1: This is the first param.
            param2: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """
        return self.positions

    def getVelocities(self):
        """
        Opdaterer a

        Args:
            param1: This is the first param.
            param2: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """

        return np.sqrt((1/N)*(sum(self.velocities[0]**2)+sum(self.velocities[1]**2)))

    def getStates(self):
        return NotImplemented
        """
            Opdaterer a
    
            Args:
                param1: This is the first param.
                param2: This is a second param.
    
            Returns:
                This is a description of what is returned.
    
            Raises:
                KeyError: Raises an exception.
            """


celle = GasCell(start_velocities=start_velocities,
                start_positions=start_positions,
                N=N)

# Laver figur og animation
fig, ax = plt.subplots(2, 1, figsize=(20, 20))

ax[0].set_ylim(-d, d)
ax[0].set_xlim(-d, d)
ax[0].set_aspect("equal")
ax[1].set_xlabel("$Tid (s)$")
ax[1].set_ylabel("$v_rms (m/s)$")


dots = ax[0].scatter([], [])


length_of_animation = 8000
positioner = []
farver = []
rms = []


tid = np.arange(0, length_of_animation, 1)*0.001

for i in range(length_of_animation):
    celle.simulate()
    positioner.append(celle.getPositions().copy())
    farver.append(celle.color.copy())
    rms.append(celle.getVelocities().copy())



timer = ax[0].text(3, 8, "")

def animate(i):
    dots.set_offsets(positioner[i].T)
    dots.set_color(farver[i])
    tid_timer = i*dt
    
    timer.set_text(f"Tid: {tid_timer:0.4f} s")
    return dots, timer


ax[1].plot(tid, rms)

ani = anim.FuncAnimation(
    fig, animate, frames=length_of_animation, interval=10, blit=True)
plt.show()
