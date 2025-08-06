import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import List, Dict, Tuple, Union, Any
import copy

class TCN:
    """
    A class to represent a single or ensemble of Typed Cell Niches (TCNs).

    Args:
        run_id: List of run identifiers.
        fov_id: List of field of view identifiers.
        tcn_arr: Array representing the TCNs, shape (n_hops, n_cell_types).
        label: List of labels corresponding to each TCN (0/1).
        center_cell_index: List of indices of the center cell for each TCN.
        cell_type_indexing: Mapping from cell type name to column index.
        binary_tcn_arr: Optional binary indicator array; if None, derived from `tcn_arr`.
        marker_tcn_arr: Optional marker-specific TCN array, same shape as `tcn_arr`.
        cell_type_indeces_dict: Optional dict mapping each cell type to a dict of
            hop→list of cell indices.

    Attributes:
        run_id: List[int] — run identifiers.
        fov_id: List[int] — field of view identifiers.
        tcn_arr: np.ndarray — numeric TCN array.
        binary_tcn_arr: np.ndarray — binary TCN array.
        marker_tcn_arr: Optional[np.ndarray] — marker TCN array.
        label: List[int] — responder labels.
        center_cell_index: List[int] — center‐cell indices.
        cell_type_indexing: Dict[str, int] — cell type → column idx.
        cell_type_indeces_dict: Optional[Dict[str, Dict[int, List[int]]]].
        avrage_nei_sizes: Optional[np.ndarray] — average neighborhood sizes override.
    """

    def __init__(self, run_id: List[int], fov_id: List[int], tcn_arr: np.ndarray, label: List[int], center_cell_index: List[int], cell_type_indexing: Dict[str, int], binary_tcn_arr: None| np.ndarray = None,  marker_tcn_arr: None| np.ndarray = None, cell_type_indeces_dict: Dict[str, Dict[int, List[int]]] = None):
        """
        Initializes a TCN object, deriving binary_tcn_arr from `tcn_arr` if not provided.

        Args:
            run_id: List[int] — run identifiers.
            fov_id: List[int] — field of view identifiers.
            tcn_arr: np.ndarray — array of TCN values.
            label: List[int] — responder labels (0 or 1).
            center_cell_index: List[int] — indices of the center cell per TCN.
            cell_type_indexing: Dict[str, int] — mapping cell type → tcn_arr column.
            binary_tcn_arr: Optional[np.ndarray] — binary version of `tcn_arr`.
            marker_tcn_arr: Optional[np.ndarray] — marker-specific array.
            cell_type_indeces_dict: Optional[Dict[str, Dict[int, List[int]]]] — 
                per‐cell‐type hop‐index mapping.

        """
        self.run_id = run_id
        self.fov_id = fov_id
        self.tcn_arr = tcn_arr
        self.label = label
        self.center_cell_index = center_cell_index
        self.cell_type_indexing = cell_type_indexing
        self.avrage_nei_sizes = None
        self.marker_tcn_arr = marker_tcn_arr
        self.cell_type_indeces_dict = cell_type_indeces_dict
        if binary_tcn_arr is not None:
            self.binary_tcn_arr = binary_tcn_arr
        else:
            self.binary_tcn_arr = (self.tcn_arr > 0).astype(int)

    def __add__(self, other: 'TCN') -> 'TCN':
        """
        Adds two TCN objects element‐wise, concatenating metadata lists.

        Args:
            other: TCN — another TCN object with identical `tcn_arr` shape.

        Returns:
            TCN — new object with summed arrays and concatenated lists.

        Raises:
            ValueError: if `self.tcn_arr.shape != other.tcn_arr.shape`.
        """
        if self.tcn_arr.shape != other.tcn_arr.shape:
            raise ValueError("TCN arrays must have the same shape to be added.")

        run_id = self.run_id + other.run_id
        fov_id = self.fov_id + other.fov_id
        tcn_arr = self.tcn_arr + other.tcn_arr
        label = self.label + other.label
        center_cell_index = self.center_cell_index + other.center_cell_index
        binary_tcn_arr = self.binary_tcn_arr + other.binary_tcn_arr
        marker_tcn_arr = self.marker_tcn_arr + other.marker_tcn_arr
        combined_cell_type_indeces_dict = copy.copy(self.cell_type_indeces_dict)
        for key in self.cell_type_indeces_dict:
            for hop in self.cell_type_indeces_dict[key]:
                combined_cell_type_indeces_dict[key][hop] += other.cell_type_indeces_dict[key][hop]
        tcn2return = TCN(run_id, fov_id, tcn_arr, label, center_cell_index, self.cell_type_indexing, binary_tcn_arr, marker_tcn_arr, combined_cell_type_indeces_dict)
        if self.avrage_nei_sizes is not None and other.avrage_nei_sizes is not None:
            tcn2return.set_avarage_neighborhood_sizeS(self.avrage_nei_sizes)
        return tcn2return

    def __sub__(self, other: 'TCN') -> 'TCN':
        """
        Subtracts another TCN object element‐wise, concatenating metadata lists.

        Args:
            other: TCN — another TCN object with identical `tcn_arr` shape.

        Returns:
            TCN — new object with subtracted arrays and concatenated lists.

        Raises:
            ValueError: if `self.tcn_arr.shape != other.tcn_arr.shape`.
        """
        if self.tcn_arr.shape != other.tcn_arr.shape:
            raise ValueError("TCN arrays must have the same shape to be subtracted.")

        run_id = self.run_id + other.run_id
        fov_id = self.fov_id + other.fov_id
        tcn_arr = self.tcn_arr - other.tcn_arr
        label = self.label + other.label
        center_cell_index = self.center_cell_index + other.center_cell_index
        binary_tcn_arr = self.binary_tcn_arr - other.binary_tcn_arr
        matcher_tcn_arr = self.marker_tcn_arr - other.marker_tcn_arr
        combined_cell_type_indeces_dict = copy.copy(self.cell_type_indeces_dict)
        for key in self.cell_type_indeces_dict:
            for hop in self.cell_type_indeces_dict[key]:
                combined_cell_type_indeces_dict[key][hop] += other.cell_type_indeces_dict[key][hop]
        tcn2return = TCN(run_id, fov_id, tcn_arr, label, center_cell_index, self.cell_type_indexing, binary_tcn_arr, matcher_tcn_arr, combined_cell_type_indeces_dict)
        if self.avrage_nei_sizes is not None and other.avrage_nei_sizes is not None:
            tcn2return.set_avarage_neighborhood_sizeS(self.avrage_nei_sizes)
        return tcn2return

    def get_cell_type_subset_representation(self, cell_types: List[str]) -> np.ndarray:
        """
        Returns the numeric TCN array restricted to a subset of cell types.

        Args:
            cell_types: List[str] — cell type names to include.

        Returns:
            np.ndarray — shape (n_hops, len(cell_types)) subset of `tcn_arr`.

        Raises:
            KeyError: if any `cell_type` is not in `cell_type_indexing`.
        """
        indices = [self.cell_type_indexing[ct] for ct in cell_types]
        tcn_arr_subset = self.tcn_arr[:, indices]
        return tcn_arr_subset
    
    def get_cell_type_subset_binary_representation(self, cell_types: List[str]) -> np.ndarray:
        """
        Returns the binary TCN array restricted to a subset of cell types.

        Args:
            cell_types: List[str] — cell type names to include.

        Returns:
            np.ndarray — shape (n_hops, len(cell_types)) subset of `binary_tcn_arr`.

        Raises:
            KeyError: if any `cell_type` is not in `cell_type_indexing`.
        """
        indices = [self.cell_type_indexing[ct] for ct in cell_types]
        binary_tcn_arr_subset = self.binary_tcn_arr[:, indices]
        return binary_tcn_arr_subset
    
    @staticmethod
    def normalze_tcn_arr(tcn_arr: np.ndarray) -> np.ndarray:
        """
        Row‐normalizes a TCN array so each row sums to 1 (zero‐sum rows left unchanged).

        Args:
            tcn_arr: np.ndarray — input array, shape (n_hops, n_cell_types).

        Returns:
            np.ndarray — row‐normalized array, same shape as input.
        """
        row_sums = tcn_arr.sum(axis=1)
        row_sums[row_sums == 0] = 1
        row_sums = row_sums.reshape(-1, 1)
        tcn_arr_normed = tcn_arr / row_sums
        return tcn_arr_normed

    def get_normed_representation(self) -> np.ndarray:
        """
        Returns a row-normalized representation of the TCN array.

        Returns:
            np.ndarray — the normalized TCN array where each row sums to 1.
        """
        return self.normalze_tcn_arr(self.tcn_arr)
    
    def get_mean_representation(self) -> np.ndarray:
        """
        Returns the mean representation of the TCN array over runs,
        optionally adjusted by average neighborhood sizes.

        Returns:
            np.ndarray — mean TCN array, shape (n_hops, n_cell_types).
        """
        mean_rep = self.tcn_arr / len(self.run_id)
        if self.avrage_nei_sizes is not None:
            mean_rep = mean_rep / self.avrage_nei_sizes.reshape(-1, 1)
        return mean_rep
    
    def get_mean_binary_representation(self) -> np.ndarray:
        """
        Returns the mean binary representation of the TCN array over runs.

        Returns:
            np.ndarray — mean binary TCN array, values in [0, 1].
        """
        return self.binary_tcn_arr / len(self.run_id)
    

    def get_remap_cell_types_tcn_object(self, meta_group_mapping: Dict[str, None | str]) -> 'TCN':
        """
        Aggregates and remaps cell types according to provided meta-group mapping.

        Args:
            meta_group_mapping: Dict[str, Optional[str]] —
                mapping from original cell type to new meta-group;
                if value is None, that cell type is removed.

        Returns:
            TCN — new object with aggregated and remapped cell types.

        Raises:
            ValueError: if all cell types are mapped to None, leaving no groups.
        """
        # check that not all cell types are removed
        if all([v is None for v in meta_group_mapping.values()]):
            raise ValueError("All cell types are removed. Please check the meta_group_mapping.")
        new_cell_type_indexing = {}
        runnning_index = 0
        for meta_group in meta_group_mapping.values():
            if meta_group is not None and meta_group not in new_cell_type_indexing:
                new_cell_type_indexing[meta_group] = runnning_index
                runnning_index += 1
        new_tcn_arr = np.zeros((self.tcn_arr.shape[0], len(new_cell_type_indexing)))
        new_binary_tcn_arr = np.zeros((self.binary_tcn_arr.shape[0], len(new_cell_type_indexing)))
        for cell_type, index in self.cell_type_indexing.items():
            meta_group = meta_group_mapping.get(cell_type, None)
            if meta_group is not None:
                new_tcn_arr[:, new_cell_type_indexing[meta_group]] += self.tcn_arr[:, index]
                new_binary_tcn_arr[:, new_cell_type_indexing[meta_group]] += self.binary_tcn_arr[:, index]
        tcn_return =  TCN(self.run_id, self.fov_id, new_tcn_arr, self.label, self.center_cell_index, new_cell_type_indexing, new_binary_tcn_arr)
        tcn_return.set_avarage_neighborhood_sizeS(self.avrage_nei_sizes)
        return tcn_return


    def is_homogenous_tcn_in_type(self, cell_types: List[str]) -> bool:
        """
        Checks if the TCN contains only the specified cell types.

        Args:
            cell_types: List[str] — cell types to check for homogeneity.

        Returns:
            bool — True if all other cell-type columns are zero.
        """
        indices_not_included = [self.cell_type_indexing[ct] for ct in self.cell_type_indexing.keys() if ct not in cell_types]
        return np.all(self.tcn_arr[:, indices_not_included] == 0)
        
    def plot_pie_chart(self, cell_types_to_plot: List[str], hops_to_plot: int = 0, type: str = 'normed') -> None:
        """
        Plots a pie chart of the specified cell types at a given hop.

        Args:
            cell_types_to_plot: List[str] — names of cell types to include.
            hops_to_plot: int — hop index to plot (row of the array).
            type: str — representation mode; one of
                'normed', 'raw', 'mean', 'mean_binary'.

        Raises:
            ValueError: if `type` is not one of the allowed modes.
        """
        indices_to_plot = [self.cell_type_indexing[ct] for ct in cell_types_to_plot]
        tcn = self._get_representation_by_type(type)
        data_filt = tcn[hops_to_plot, indices_to_plot]
        # filter out 0 values
        data_to_plot = data_filt[data_filt > 0]
        cell_types_to_plot = [cell_types_to_plot[i] for i in range(len(cell_types_to_plot)) if data_filt[i] > 0]
        # use pastel colors in the pie chart instead of the default colors
        plt.pie(data_to_plot, labels=cell_types_to_plot, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        # plt.title(f'Cell Type Distribution at Hop {hops_to_plot}, {type}')
        plt.show()

    def is_tcn_center_cell_in_types(self, cell_types: List[str]) -> bool:
        """
        Checks if the center cell belongs to any of the specified cell types.

        Args:
            cell_types: List[str] — cell types to check.

        Returns:
            bool — True if the center cell has a positive value in any of those types.
        """
        center_indices = [self.cell_type_indexing[ct] for ct in cell_types]
        return np.any(self.tcn_arr[0, center_indices] > 0)
        
    def _get_patient_labels(self) -> Dict[Tuple[int, int], int]:
        """
        Retrieves the unique patient labels based on (run_id, fov_id).

        Returns:
            Dict[Tuple[int, int], int] — mapping from (run_id, fov_id) to label.
        """
        run_id_fov_id = list(zip(self.run_id, self.fov_id))
        unique_indices = np.unique(run_id_fov_id, axis=0, return_index=True)[1]
        patient_labels = {run_id_fov_id[i]: self.label[i] for i in unique_indices}
        return patient_labels
        

    def get_patient_histogram_numbers(self) -> Tuple[int, int]:
        """
        Returns a histogram of how many responders and non-responders correspond to the TCN object.

        Returns:
            Tuple[int, int] — (n_responders, n_non_responders).
        """
        patient_labels = self._get_patient_labels()
        unique_labels = np.array(list(patient_labels.values()))
        return np.sum(unique_labels), len(unique_labels) - np.sum(unique_labels)

    def get_center_cells_histogram_numbers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Counts how many center cells are responders vs non-responders.

        Returns:
            Tuple[int, int] — (n_responder_cells, n_non_responder_cells).
        """
        return np.sum(self.label), len(self.label) - np.sum(self.label)
            
    def plot_center_cells_histogram(self) -> None:
        """
        Plots a bar chart of responder vs non-responder center cells.
        """
        responder_cells, non_responder_cells = self.get_center_cells_histogram_numbers()
        plt.bar(["Responders", "Non-Responders"], [responder_cells, non_responder_cells], color=['#F5A9A9', '#A9D0F5'])
        # plt.title("Histogram of Responder and Non-Responder Cells")
        plt.show()

    def get_idws_histogram(self, idws_scores: Dict[Tuple[int, int], Dict[int, float]], pos_thresh: float = 0.5, neg_thersh: float = -0.5) -> np.ndarray:
        """
        Binarizes IDWS scores and counts responders vs non-responders.

        Args:
            idws_scores: mapping from (run_id, fov_id) to per-cell scores.
            pos_thresh: float — scores above this count as responder.
            neg_thersh: float — scores below this count as non-responder.

        Returns:
            Tuple[int, int] — (n_responder_cells, n_non_responder_cells).

        Raises:
            AssertionError: if pos_thresh <= neg_thersh.
            KeyError: if (run_id, fov_id) or cell index is missing in idws_scores.
        """
        assert pos_thresh > neg_thersh
        responder_cells = 0
        non_responder_cells = 0
        for run_id, fov_id, cell_id in zip(self.run_id, self.fov_id, self.center_cell_index):
            if idws_scores[(run_id, fov_id)][cell_id] > pos_thresh:
                responder_cells += 1
            elif idws_scores[(run_id, fov_id)][cell_id] < neg_thersh:
                non_responder_cells += 1
        return responder_cells, non_responder_cells
    
    
    def plot_idws_histogram(self, idws_scores: Dict[int, float], pos_thresh: float = 0.5, neg_thersh: float = -0.5) -> None:
        """
        Plots a bar chart of responder vs non-responder cells based on IDWS scores.

        Args:
            idws_scores: mapping from (run_id, fov_id) to per-cell scores.
            pos_thresh: float — positive threshold (default 0.5).
            neg_thersh: float — negative threshold (default -0.5).
        """
        responder_cells, non_responder_cells = self.get_idws_histogram(idws_scores, pos_thresh, neg_thersh)
        plt.bar(["Responders", "Non-Responders"], [responder_cells, non_responder_cells], color=['red', 'blue'])
        # plt.title("Histogram of Responder and Non-Responder IWDS Cells' scores")
        plt.show()

    def plot_tcn_heatmap(self, cell_types_to_remove: None | List[str] = None, remove_center: bool = True, type: str = 'normed') -> None:
        """
        Displays a heatmap of the TCN representation.

        Args:
            cell_types_to_remove: Optional[List[str]] — names to exclude.
            remove_center: bool — if True, omit the center cell row.
            type: str — representation mode; one of
                'normed', 'raw', 'mean', 'mean_binary'.

        Raises:
            ValueError: if `type` is not one of the allowed modes.
        """
        fig, ax = plt.subplots()  

        tcn = self.tcn_arr
        if remove_center:
            tcn = tcn[1:]
            column_add = 1
        else:
            column_add = 0 

        if cell_types_to_remove is not None:
            indices_to_remove = [self.cell_type_indexing.get(ct, -1) for ct in cell_types_to_remove]
            indices_to_remove = [i for i in indices_to_remove if i != -1]
        tcn = self._get_representation_by_type(type)
        if cell_types_to_remove is not None:
            tcn = np.delete(tcn, indices_to_remove, axis=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"], N=256)
        # plot the heatmap
        ax.imshow(tcn[::-1], cmap=cmap)
        ax.set_xticks(np.arange(tcn.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(tcn.shape[0] + (1-column_add)) - .5, minor=True)
        ax.tick_params(which='minor', size=0)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.set_xticks(np.arange(tcn.shape[1]))
        ax.set_yticks(np.arange(tcn.shape[0]) - column_add)
        
        # Setting labels
        cell_types_named = [ct for ct in self.cell_type_indexing if ct not in cell_types_to_remove] if cell_types_to_remove else list(self.cell_type_indexing.keys())
        ax.set_xticklabels(cell_types_named, rotation=90)
        ax.set_yticklabels(np.arange(tcn.shape[0], 0, -1))
        ax.set_ylabel("Hops")
        ax.set_xlabel("Cell Types")
        # plt.title(f'TCN Heatmap {type}')
        plt.show()

    def _get_representation_by_type(self, type: str) -> np.ndarray:
        """
        Retrieves a TCN array representation based on the specified mode.

        Args:
            type: str — one of 'normed', 'raw', 'mean', 'mean_binary'.

        Returns:
            np.ndarray — the corresponding TCN array.

        Raises:
            ValueError: if `type` is not recognized.
        """
        if type == 'normed':
            return self.get_normed_representation()
        elif type == 'mean':
            return self.get_mean_representation()
        elif type == 'mean_binary':
            return self.get_mean_binary_representation()
        elif type == 'raw':
            return self.tcn_arr
        else:
            raise ValueError(f"Invalid type {type}. Must be 'normed' or 'raw' or 'mean' or 'mean_binary'")

    def plot_patient_histograms(self):
        """
        Plots responder vs non-responder patient counts as a bar chart.
        """
        responders, non_responders = self.get_patient_histogram_numbers()
        plt.bar(["Responders", "Non-Responders"], [responders, non_responders], color=['#F5A9A9', '#A9D0F5']) 
        # plt.title("Histogram of Responders and Non-Responders")
        plt.show()

    def get_patients_counts(self) -> Dict[Tuple[int, int], int]:
        """
        Counts the number of TCNs associated with each patient.

        Returns:
            Dict[Tuple[int, int], int] — mapping (run_id, fov_id) → count of TCNs.
        """
        run_id_fov_id = list(zip(self.run_id, self.fov_id))
        patient_counts = dict()
        for run_id, fov_id in run_id_fov_id:
            patient_counts[(run_id, fov_id)] = patient_counts.get((run_id, fov_id), 0) + 1
        return patient_counts
    
    def plot_patient_counts_with_labels(self, title: str = "Number of TCNs Assosiated with Each Patient", enumarate_patients: bool = True):
        """
        Plots TCN counts per patient, colored by responder status.

        Args:
            title: str — chart title.
            enumarate_patients: bool — if True, use numeric indices on x-axis,
                else use (run_id, fov_id) tuples.
        """
        patient_counts = self.get_patients_counts()
        patient_labels = self._get_patient_labels()
        colors = ['red' if patient_labels[patient] == 1 else 'blue' for patient in patient_counts.keys()]
        plt.bar(range(len(patient_counts)), list(patient_counts.values()), color=colors)
        # put fov_id on the x-axis
        if enumarate_patients:
            plt.xticks(range(len(patient_counts)), range(len(patient_counts)), rotation=90)
        else:
            plt.xticks(range(len(patient_counts)), list(patient_counts.keys()), rotation=90)
        # plt.title(title)
        plt.show()
        

    def get_binary_representation(self) -> np.ndarray:
        """
        Returns a binary indicator array of the TCN values (>0).

        Returns:
            np.ndarray — binary TCN array, same shape as `tcn_arr`.
        """
        return (self.tcn_arr > 0).astype(int)
    
    def set_avarage_neighborhood_sizeS(self, avrage_nei_sizes: np.ndarray) -> None:
        """
        Sets an override for average neighborhood sizes, used by `get_mean_representation`.

        Args:
            avrage_nei_sizes: np.ndarray — array of length n_hops,
                average neighborhood sizes per hop.
        """
        self.avrage_nei_sizes = avrage_nei_sizes

    def get_avarage_neighborhood_sizes(self) -> np.ndarray:
        """
        Retrieves or computes average neighborhood sizes per hop.

        Returns:
            np.ndarray — array of length n_hops with average sizes.
        """
        if self.avrage_nei_sizes is None:
            total_num_of_cells = np.sum(self.tcn_arr, axis=1)
            mean_nei_sizes = total_num_of_cells / len(self.run_id)
            return mean_nei_sizes
        return self.avrage_nei_sizes
