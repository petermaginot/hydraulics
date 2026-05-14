
def phase_env():
    import matplotlib.pyplot as plt
    AS_g = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    try:
        AS_g.build_phase_envelope("dummy")
        PE = AS_g.get_phase_envelope_data()
        plt.plot(PE.T, PE.p, '-', label='Composition')
        plt.xlabel('Temperature [K]')
    except ValueError as VE:
        print(VE)

    plt.ylabel('Pressure [Pa]')
    plt.yscale('log')
    plt.title('Phase Envelope for Selected Composition')
    plt.legend(loc='lower right', shadow=True)
    plt.savefig('methane-ethane.png')

