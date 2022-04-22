from typing import Tuple, List, Union, Optional, Any
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, QuantumCircuit

def show_state(
    statevector: Union[QuantumCircuit, np.ndarray],
    amp_norm: Optional[Tuple[float, str]] = None,
    phase_norm: Tuple[float, str] = (np.pi, '\pi'),
    global_phase: Optional[Union[float, str]] = None,
    register_sizes: Optional['array_like'] = None,
    terms_per_row: int = 8,
    binary: bool = False,
    epsilon: float = 1.e-3,
    state_label: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None
) -> Union[None, mpl.figure.Figure]:
    """Show the quantum state of the circuit as text in a matplotlib Figure.
    
    Args:
        statevector: Input statevector or a QuantumCircuit whose final state is to be extracted.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in latex).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in latex).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        register_sizes: Specification of the sizes of registers as a list of ints.
        terms_per_row: Number of terms to show per row.
        binary: Show ket indices in binary.
        state_label: If not None, prepend '|`state_label`> = ' to the printout.
        ax: Axes object. A new axes is created if None.
        
    Returns:
        The newly created figure object if ax is None.
    """
    lines = statevector_expr(statevector, amp_norm=amp_norm, phase_norm=phase_norm,
                             global_phase=global_phase, register_sizes=register_sizes,
                             terms_per_row=terms_per_row, binary=binary, epsilon=epsilon,
                             state_label=state_label)
    
    if isinstance(lines, str):
        lines = [lines]

    row_texts = list(f'${line}$' for line in lines)
    
    fig = None
    if ax is None:
        fig = plt.figure(figsize=[10., 0.5 * len(lines)])
        ax = fig.add_subplot()
        
    ax.axis('off')
    
    num_rows = len(row_texts)
    
    for irow, row_text in enumerate(row_texts):
        ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), row_text, fontsize='x-large', ha='center')

    if fig is not None:
        return fig
    
    
def statevector_expr(
    statevector: Union[QuantumCircuit, np.ndarray],
    amp_norm: Optional[Tuple[float, str]] = None,
    phase_norm: Tuple[float, str] = (np.pi, '\pi'),
    global_phase: Optional[Union[float, str]] = None,
    register_sizes: Optional['array_like'] = None,
    terms_per_row: int = 8,
    binary: bool = False,
    amp_format: str = '.3f',
    phase_format: str = '.2f',
    epsilon: float = 1.e-3,
    state_label: Optional[str] = r'\text{final}'
):
    ## If a QuantumCircuit is passed, extract the statevector
    
    if isinstance(statevector, QuantumCircuit):
        # Run the circuit in statevector_simulator and obtain the final state statevector
        simulator = Aer.get_backend('statevector_simulator')

        circuit = transpile(statevector, backend=simulator)
        statevector = np.asarray(simulator.run(circuit).result().data()['statevector'])
        
    if register_sizes is None:
        subsystem_dims = None
    else:
        subsystem_dims = tuple(2 ** s for s in reversed(register_sizes))
        
    expr = QuantumObjectExpression(statevector,
                                   amp_norm=amp_norm,
                                   phase_norm=phase_norm,
                                   global_phase=global_phase,
                                   subsystem_dims=subsystem_dims,
                                   terms_per_row=terms_per_row,
                                   binary=binary,
                                   amp_format=amp_format,
                                   phase_format=phase_format,
                                   epsilon=epsilon,
                                   lhs_label=state_label)
    
    lines = expr.compose()
    
    # For backwards compatibility
    if terms_per_row == 0:
        return lines[0]
    else:
        return lines


class QuantumObjectExpression:
    """Helper class to compose a LaTeX expression of a given quantum object.

    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        subsystem_dims: Specification of the dimensions of the subsystems.
        terms_per_row: Number of terms to show per row.
        binary: Show bra and ket indices in binary.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend '|`state_label`> = ' to the printout.
    """
    
    def __init__(
        self,
        qobj: Any,
        amp_norm: Optional[Tuple[float, str]] = None,
        phase_norm: Tuple[float, str] = (np.pi, '\pi'),
        global_phase: Optional[Union[float, str]] = None,
        subsystem_dims: Optional['array_like'] = None,
        terms_per_row: int = 0,
        binary: bool = False,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Union[str, None] = r'\text{final}'
    ):
        self.qobj = qobj
        self.amp_norm = amp_norm
        self.phase_norm = phase_norm
        self.global_phase = global_phase
        self.subsystem_dims = subsystem_dims
        self.terms_per_row = terms_per_row
        self.binary = binary
        self.amp_format = amp_format
        self.phase_format = phase_format
        self.epsilon = epsilon       

    def compose(self) -> List[str]:
        """Compose a LaTeX expression as a list of lines."""
        
        ket, bra, oper = range(3)
        if len(self.qobj.shape) == 1 or self.qobj.shape[1] == 1:
            objtype = ket
            dim = self.qobj.shape[0]
        elif self.qobj.shape[0] == 1 and self.qobj.shape[1] != 1:
            objtype = bra
            dim = self.qobj.shape[1]
        else:
            objtype = oper
            dim = self.qobj.shape[0] # Only limiting to square matrices

        subsystem_dims = self.subsystem_dims
        if subsystem_dims is None:
            subsystem_dims = np.array([dim])
            
        assert np.prod(subsystem_dims) == dim, (f'Product of subsystem dimensions {np.prod(subsystem_dims)}'
                                                f' and qobj dimension {dim} do not match')

        # State label format template
        if self.binary:
            log2_dims = np.log2(np.asarray(subsystem_dims))
            assert np.allclose(log2_dims, np.round(log2_dims))
            label_template = ':'.join(f'{{:0{s}b}}' for s in log2_dims.astype(int))
        else:
            label_template = ':'.join(['{}'] * len(subsystem_dims))

        # Amplitude format template
        amp_template = f'{{:{self.amp_format}}}'

        # Phase format template
        phase_template = f'{{:{self.phase_format}}}'

        ## Preprocess the qobj

        # Absolute value and phase of the amplitudes
        if hasattr(self.qobj, 'data'):
            # Sparse matrix
            absamp = np.abs(self.qobj.data)
            phase = np.angle(self.qobj.data)
        else:
            absamp = np.abs(self.qobj)
            phase = np.angle(self.qobj)

        # Normalize the abs amplitudes and identify integral values
        if self.amp_norm is not None:
            absamp /= self.amp_norm[0]
            
        rounded_amp = np.round(absamp).astype(int)
        amp_is_int = np.isclose(rounded_amp, absamp, rtol=self.epsilon)
        rounded_amp = np.where(amp_is_int, rounded_amp, -1)
        
        # Shift and normalize the phases and identify integral values
        phase_offset = 0.
        if self.global_phase is not None:
            if self.global_phase == 'mean':
                phase_offset = np.mean(phase)
            else:
                phase_offset = self.global_phase
                
            phase -= phase_offset

        twopi = 2. * np.pi
        
        while np.any((phase < 0.) | (phase >= twopi)):
            phase = np.where(phase >= 0., phase, phase + twopi)
            phase = np.where(phase < twopi, phase, phase - twopi)

        def normalize_phase(phase):
            reduced_phase = phase / (np.pi / 2.)
            axis_proj = np.round(reduced_phase).astype(int)
            on_axis = np.isclose(axis_proj, reduced_phase, rtol=0., atol=self.epsilon)
            axis_proj = np.where(on_axis, axis_proj, -1)

            if self.phase_norm is not None:
                phase /= self.phase_norm[0]

            rounded_phase = np.round(phase).astype(int)
            phase_is_int = np.isclose(rounded_phase, phase, rtol=0., atol=self.epsilon)
            rounded_phase = np.where(phase_is_int, rounded_phase, -1)
            
            return phase, axis_proj, rounded_phase
        
        phase, axis_proj, rounded_phase = normalize_phase(phase)

        ## Compose the LaTeX expressions
        
        # Show only terms with absamp < absamp * epsilon
        amp_atol = np.amax(absamp) * self.epsilon
        amp_is_zero = np.isclose(np.zeros_like(absamp), absamp, atol=amp_atol)
        term_indices = list(zip(*np.logical_not(amp_is_zero).nonzero())) # convert into list of tuples
        
        # Make tuples of quantum state labels and format the term indices
        if isinstance(self.qobj, scipy.sparse.csr_matrix):
            # CSR matrix: diff if indptr = number of elements in each row
            repeats = np.diff(self.qobj.indptr)
            row_labels_flat = np.repeat(np.arange(self.qobj.shape[0]), repeats)
            # unravel into row indices accounting for the tensor product
            if objtype in (ket, oper):
                row_labels = np.unravel_index(row_labels_flat, subsystem_dims)
            if objtype in (bra, oper):
                col_labels = np.unravel_index(self.qobj.indices, subsystem_dims)
                
        elif isinstance(self.qobj, np.ndarray):
            if objtype in (ket, oper):
                row_labels = np.unravel_index(np.arange(dim), subsystem_dims)
            if objtype in (bra, oper):
                col_labels = np.unravel_index(np.arange(dim), subsystem_dims)
                
        else:
            raise NotImplementedError('Unsupported qobj type')
            
        def phase_expr(phase, axis_proj, rounded_phase):
            if axis_proj == -1:
                # Not on Re or Im axis
                expr = 'e^{'

                if rounded_phase == -1:
                    expr += phase_template.format(phase)
                elif rounded_phase != 1:
                    expr += f'{rounded_phase:d}'

                if self.phase_norm is not None:
                    if rounded_phase != 1:
                        expr += r' \cdot '
                    expr += self.phase_norm[1]

                expr += ' i}'

            else:
                expr = ''

                if axis_proj >= 2:
                    expr += '-'
                    
                if axis_proj % 2 == 1:
                    expr += 'i'

            return expr
            
        # List to be concatenated into the final latex string
        lines = []
        str_terms = []

        # Pre- and Post-expressions
        pre_expr = ''
        post_expr = ''
        
        if self.amp_norm is not None:
            pre_expr += self.amp_norm[1]

        if phase_offset != 0.:
            norm_offset, offset_proj, rounded_offset = normalize_phase(phase_offset)
            pre_expr += phase_expr(norm_offset, offset_proj, rounded_offset)

        if pre_expr:
            pre_expr += r'\left('
            post_expr += r'\right)'

        # Stringify each term
        for idx in term_indices:
            phase_factor = phase_expr(phase[idx], axis_proj[idx], rounded_phase[idx])
                
            unsigned_coeff = ''
            if rounded_amp[idx] == -1:
                unsigned_coeff += amp_template.format(absamp[idx])
            elif rounded_amp[idx] != 1:
                unsigned_coeff += f'{rounded_amp[idx]:d}'

            term_qobj = ''
            if objtype in (ket, oper):
                term_qobj += '| ' + ','.join(f'{r[idx[0]]}' for r in row_labels) + r' \rangle'
            if objtype in (bra, oper):
                if len(idx) == 2:
                    ic = idx[1]
                else:
                    ic = idx[0]
                    
                term_qobj += r'\langle ' + ','.join(f'{c[ic]}' for c in col_labels) + '|'
                
            # Sign of this term
            if phase_factor.startswith('-'):
                sign = ' - '
                phase_factor = phase_factor[1:]
            elif str_terms or lines:
                sign = ' + '
            else:
                sign = ''

            str_terms.append(sign + unsigned_coeff + phase_factor + term_qobj)

            if self.terms_per_row > 0 and len(str_terms) == terms_per_row:
                # The line is full - concatenate the terms and append to the line list
                lines.append(''.join(str_terms))
                str_terms = []

        if len(str_terms) != 0:
            lines.append(''.join(str_terms))
            
        if pre_expr:
            lines[0] = pre_expr + lines[0]
            lines[-1] += post_expr

            if len(lines) > 1:
                lines[0] += r'\right.'
                lines[-1] = r'\left. ' + lines[-1]

        return lines

    def _repr_latex_(self) -> str:
        lines = self.compose()
        if len(lines) == 0:
            return ''
        elif len(lines) == 1:
            return rf'$\displaystyle {lines[0]} $'
        else:
            return r'\begin{split} ' + r' \\ '.join(lines) + ' \end{split}'