import random
import numpy as np
import tensorflow as tf

from .audio_tse import WaveReader
from .conf_unet_tse_32ms import dev_data, train_data


def numpy_collate(batch):
    """
    Matches PyTorch's default_collate by stacking items into batched NumPy arrays.
    This strictly avoids TensorFlow eager context/GPU allocation crashes during data loading.
    """
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(batch[0][key], np.ndarray):
            # Stack arrays along a new batch dimension (axis 0)
            collated[key] = np.stack(values, axis=0)
        elif isinstance(batch[0][key], (int, float, np.number)):
            collated[key] = np.array(values)
        else:
            collated[key] = values
    return collated


def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=80000,
                    batch_size=4):
    dataset = Dataset(**(data_kwargs or {}))
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


def get_spk_ivec(key):
    '''
      409o030h_1.7445_029o0304_-1.7445_409c0211
    '''
    spk = key.split('_')[-1][0:3]
    print(spk)


class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, aux_scp=None, sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]
        aux = self.aux[key]
           
        return {
            "mix": mix.astype(np.float32),
            "ref": ref.astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "key": key
        }


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=2000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = chunk["aux"].shape[0]
        chunk["valid_len"] = int(self.chunk_size)
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        if N < self.least:
            return []
        chunks = []
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunks.append(chunk)
        else:
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT using pure Python/NumPy
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=4,
                 train=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)

    def _get_eg_batches(self):
        indices = list(range(len(self.dataset)))
        if self.train:
            random.shuffle(indices)
            
        eg_batch_size = max(1, self.batch_size // 2)
        for i in range(0, len(indices), eg_batch_size):
            batch_indices = indices[i:i + eg_batch_size]
            batch_egs = [self.dataset[idx] for idx in batch_indices]
            chunks = []
            for eg in batch_egs:
                chunks += self.splitter.split(eg)
            yield chunks

    def _pad_aux(self, chunk_list):
        lens_list = [chunk_item['aux_len'] for chunk_item in chunk_list]
        max_len = np.max(lens_list)
        
        for idx in range(len(chunk_list)):
            P = max_len - len(chunk_list[idx]["aux"])
            chunk_list[idx]["aux"] = np.pad(chunk_list[idx]["aux"], (0, P), "constant")

        return chunk_list

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch as NumPy arrays
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            sub_list = self._pad_aux(chunk_list[s:s + self.batch_size])
            # Use NumPy collation to avoid TF GPU memory crashes here
            batch = numpy_collate(sub_list)
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self._get_eg_batches():
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj


if __name__ == '__main__':
    chunk_size = 32000
    train = True
    least = chunk_size // 2
    splitter = ChunkSplitter(chunk_size, train, least)  
    data = Dataset(**train_data)
    
    dataload = DataLoader(data)
    temp = []
    
    for i, obj in enumerate(dataload):
        # NOW we safely convert to TensorFlow tensors in the main loop
        mix = tf.convert_to_tensor(obj["mix"])
        ref = tf.convert_to_tensor(obj["ref"])
        aux = tf.convert_to_tensor(obj["aux"])
        
        temp.append(mix)
        print(f"Batch {i} loaded. Mix shape: {mix.shape}")
        print(f"Batch {i} loaded. Ref shape: {ref.shape}")
        print(f"Batch {i} loaded. Aux shape: {aux.shape}")
        
        if i == 2:
            break
            
    print(f"Total test batches appended: {len(temp)}")