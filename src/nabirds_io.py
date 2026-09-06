"""Python 3 port of the loaders shipped in `nabirds/nabirds.py` (bản gốc là Python 2).

Giữ nguyên tên hàm và cấu trúc trả về của script gốc để dễ đối chiếu, chỉ sửa:
  - `map(int, ...)` -> `list` thật (Python 3 trả về iterator, dùng 1 lần là hết)
  - image_id / class_id giữ dạng `str` đúng như bản gốc
  - thêm `encoding='utf-8'` (photographers.txt có ký tự non-ASCII)

Bổ sung thêm các tiện ích mà script gốc không có nhưng bài toán cần:
  - `build_label_index`  : 555 class_id lá  -> nhãn liên tục 0..554
  - `build_taxonomy`     : ánh xạ lá -> species(404) / family(228) / order(22)
"""
import os

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nabirds')


# --------------------------------------------------------------------------
# Loaders — bám sát nabirds/nabirds.py
# --------------------------------------------------------------------------
def load_bounding_box_annotations(dataset_path=DEFAULT_ROOT):
    bboxes = {}
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if not pieces:
                continue
            bboxes[pieces[0]] = [int(v) for v in pieces[1:]]
    return bboxes


def load_part_annotations(dataset_path=DEFAULT_ROOT):
    parts = {}
    with open(os.path.join(dataset_path, 'parts', 'part_locs.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if not pieces:
                continue
            image_id = pieces[0]
            parts.setdefault(image_id, [None] * 11)
            parts[image_id][int(pieces[1])] = [int(v) for v in pieces[2:]]
    return parts


def load_part_names(dataset_path=DEFAULT_ROOT):
    names = {}
    with open(os.path.join(dataset_path, 'parts', 'parts.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                names[int(pieces[0])] = ' '.join(pieces[1:])
    return names


def load_class_names(dataset_path=DEFAULT_ROOT):
    names = {}
    with open(os.path.join(dataset_path, 'classes.txt'), encoding='utf-8') as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                names[pieces[0]] = ' '.join(pieces[1:])
    return names


def load_image_labels(dataset_path=DEFAULT_ROOT):
    labels = {}
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                labels[pieces[0]] = pieces[1]
    return labels


def load_image_paths(dataset_path=DEFAULT_ROOT, path_prefix=''):
    paths = {}
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                paths[pieces[0]] = os.path.join(path_prefix, pieces[1])
    return paths


def load_image_sizes(dataset_path=DEFAULT_ROOT):
    sizes = {}
    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                sizes[pieces[0]] = [int(pieces[1]), int(pieces[2])]
    return sizes


def load_hierarchy(dataset_path=DEFAULT_ROOT):
    parents = {}
    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                parents[pieces[0]] = pieces[1]
    return parents


def load_photographers(dataset_path=DEFAULT_ROOT):
    photographers = {}
    with open(os.path.join(dataset_path, 'photographers.txt'), encoding='utf-8') as f:
        for line in f:
            pieces = line.strip().split()
            if pieces:
                photographers[pieces[0]] = ' '.join(pieces[1:])
    return photographers


def load_train_test_split(dataset_path=DEFAULT_ROOT):
    train_images, test_images = [], []
    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            if not pieces:
                continue
            (train_images if int(pieces[1]) else test_images).append(pieces[0])
    return train_images, test_images


# --------------------------------------------------------------------------
# Bổ sung
# --------------------------------------------------------------------------
def build_label_index(image_labels):
    """555 class_id lá (chuỗi) -> nhãn liên tục 0..554, sắp theo class_id số tăng dần."""
    leaf_ids = sorted(set(image_labels.values()), key=int)
    return {cid: i for i, cid in enumerate(leaf_ids)}, leaf_ids


def ancestors(class_id, hierarchy):
    """Danh sách tổ tiên từ cha trực tiếp lên tới root ('0')."""
    out = []
    cur = class_id
    while cur in hierarchy:
        cur = hierarchy[cur]
        out.append(cur)
    return out


def build_taxonomy(leaf_ids, hierarchy, class_names):
    """Với mỗi class_id lá, trả về (species, order) dạng tên chuỗi.

    CẢNH BÁO về cấu trúc cây — nó KHÔNG đồng nhất về độ sâu:
      - 265 lá ở depth 4: Birds > Perching Birds > Wood-Warblers > Yellow-rumped
        Warbler > Yellow-rumped Warbler (Breeding Myrtle)
        => depth1=order, depth2=family, depth3=species, depth4=lá
      - 290 lá ở depth 3: Birds > Ducks, Geese, and Swans > Common Eider >
        Common Eider (Adult male)
        => depth1=order, depth2=SPECIES (tầng family bị bỏ qua), depth3=lá

    Vì vậy tầng depth-2 KHÔNG phải lúc nào cũng là family (199/228 node depth-2
    thực chất là species) -> không dùng nó làm mức gộp.

    Hai mức gộp đáng tin:
      - species = cha trực tiếp của lá  -> 404 node (khớp con số ~400 loài của paper)
      - order   = node ở depth 1        -> 22 node
    """
    tax = {}
    for cid in leaf_ids:
        anc = ancestors(cid, hierarchy)          # [cha, ..., root]
        species_id = anc[0]                      # cha trực tiếp = species
        order_id = anc[-2]                       # ngay dưới root
        tax[cid] = (class_names[species_id], class_names[order_id])
    return tax
