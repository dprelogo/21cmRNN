"""Functions and utilities used to format the databases."""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import quadrature
import tools21cm as t2c


def apply_uv_coverage(Box_uv, uv_bool):
    """Apply UV coverage to the data.

    Args:
        Box_uv: data box in Fourier space
        uv_bool: mask of measured baselines

    Returns:
        Box_uv
    """
    Box_uv = Box_uv * uv_bool
    return Box_uv


def compute_uv_coverage(redshifts, ncells=200, boxsize=300):
    """Computing UV coverage box for SKA antenna configuration.

    Args:
        redshifts: list of redshifts for which the UV coverage is computed.
        ncells: lsize of a grid in UV space (in pixels)
        boxsize: size of the simulation (in Mpc)

    Returns:
        uv: UV coverage box
    """
    uv = np.empty((ncells, ncells, len(redshifts)))

    for i in range(len(redshifts)):
        print(i, end=" ")
        uv[..., i], _ = t2c.noise_model.get_uv_map(
            ncells=200, z=redshifts[i], boxsize=300
        )

    return uv


def noise(seed, redshifts, uv, ncells=200, boxsize=300.0, obs_time=1000, N_ant=512):
    """Computing telescope thermal noise.

    Args:
        seed: noise seed
        redshifts: list of redshifts for each slice of UV
        uv: UV coveragebox
        ncells: size of a box in real/UV space (in pixels)
        boxsize: size of the simulation (in Mpc)
        obs_time: total observation time (in hours)
        N_ant: number of antennas in the configuration

    Returns:
        finalBox: noise in UV space
    """
    redshifts = np.append(
        redshifts, 2 * redshifts[-1] - redshifts[-2]
    )  # appending the last difference
    finalBox = np.empty(uv.shape, dtype=np.complex64)
    for i in range(uv.shape[-1]):
        depth_mhz = t2c.cosmology.z_to_nu(redshifts[i]) - t2c.cosmology.z_to_nu(
            redshifts[i + 1]
        )
        noise = t2c.noise_model.noise_map(
            ncells=ncells,
            z=redshifts[i],
            depth_mhz=depth_mhz,
            obs_time=obs_time,
            boxsize=boxsize,
            uv_map=uv[..., i],
            N_ant=N_ant,
            seed=10000 * seed + i,
        )
        noise = t2c.telescope_functions.jansky_2_kelvin(
            noise, redshifts[i], boxsize=boxsize
        ).astype(np.complex64)
        finalBox[..., i] = noise
    return finalBox


def wedge_removal(
    OMm,
    redshifts,
    HII_DIM,
    cell_size,
    Box_uv,
    chunk_length=501,
    blackman=True,
):
    """Computing horizon wedge removal. Implements "sliding" procedure
    of removing the wedge for every redshift separately.

    Args:
        OMm: Omega matter
        redshifts: list of redshifts in a lightcone
        HII_DIM: size of the HII simulation box (see `21cmFASTv3`)
        cell_size: size of a cell in Mpc
        Box_uv: box in UV space on which wedge removal is to be computed
        chunk_length: length of a sliding chunk (in number of z-slices)
        blackman: either to use Blackman-Harris taper or not

    Returns:
        Box_final: wedge-removed box in real space
    """

    def one_over_E(z, OMm):
        return 1 / np.sqrt(OMm * (1.0 + z) ** 3 + (1 - OMm))

    def multiplicative_factor(z, OMm):
        return (
            1
            / one_over_E(z, OMm)
            / (1 + z)
            * quadrature(lambda x: one_over_E(x, OMm), 0, z)[0]
        )

    MF = jnp.array([multiplicative_factor(z, OMm) for z in redshifts]).astype(
        np.float32
    )
    redshifts = jnp.array(redshifts).astype(np.float32)

    k = jnp.fft.fftfreq(HII_DIM, d=cell_size)
    k_parallel = jnp.fft.fftfreq(chunk_length, d=cell_size)
    delta_k = k_parallel[1] - k_parallel[0]
    k_cube = jnp.meshgrid(k, k, k_parallel)

    bm = jnp.abs(jnp.fft.fft(jnp.blackman(chunk_length))) ** 2
    buffer = delta_k * (jnp.where(bm / jnp.amax(bm) <= 1e-10)[0][0] - 1)
    BM = jnp.blackman(chunk_length)[jnp.newaxis, jnp.newaxis, :]

    box_shape = Box_uv.shape
    Box_final = np.empty(box_shape, dtype=np.float32)
    empty_box = jnp.zeros(k_cube[0].shape)
    Box_uv = jnp.concatenate(
        (empty_box, jnp.array(Box_uv, dtype=jnp.float32), empty_box), axis=2
    )

    for i in range(chunk_length, box_shape[-1] + chunk_length):
        t_box = Box_uv[..., i - chunk_length // 2 : i + chunk_length // 2 + 1]
        W = k_cube[2] / (
            jnp.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
            * MF[min(i - chunk_length // 2 - 1, box_shape[-1] - 1)]
            + buffer
        )
        w = jnp.logical_or(W < -1.0, W > 1.0)
        # w = cp.array(W[i + chunk_length - 1])
        if blackman == True:
            t_box = t_box * BM
        Box_final[..., i - chunk_length] = jnp.real(
            jnp.fft.ifftn(jnp.fft.fft(t_box, axis=-1) * w)
        )[
            ..., chunk_length // 2
        ]  # taking only middle slice in redshift

    return Box_final.astype(np.float32)


def BoxCar3D(data, filter=(4, 4, 4)):
    """Computing BoxCar filter on the input data.

    Args:
        data: data to filter
        filter: filter shape

    Returns:
        filtered data
    """
    if len(data.shape) != 3:
        raise AttributeError("data has to be 3D")
    if len(filter) != 3:
        raise AttributeError("filter has to be 3D")
    s = data.shape
    Nx, Ny, Nz = filter

    return jnp.einsum(
        "ijklmn->ikm",
        data[: s[0] // Nx * Nx, : s[1] // Ny * Ny, : s[2] // Nz * Nz].reshape(
            (s[0] // Nx, Nx, s[1] // Ny, Ny, s[2] // Nz, Nz)
        ),
    ) / (Nx * Ny * Nz)
