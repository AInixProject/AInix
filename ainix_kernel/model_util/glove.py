import os
from typing import Optional, List, Mapping
import numpy as np

from ainix_common.parsing.model_specific import parse_constants
from ainix_kernel.model_util.vocab import Vocab

GLOVE_DIR = "../training/opennmt/glove_dir/"
GLOVE_DIM_TO_NAME = {
    50: "glove.6B.50d.txt",
    100: "glove.6B.100d.txt",
    300: "glove.840B.300d.txt",
}
# Precomputed average of all vecs. There's commented out code at the bottom
# finding this. See also
# https://stackoverflow.com/questions/49239941/
# what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
AVG_ALL_VECS = {
    50: [-0.12920060864176852, -0.28866239452097603, -0.012248941299726332, -0.056766888363689434,
         -0.202111085482792, -0.08389026443356357, 0.3335973726965789, 0.1604514588494143,
         0.03867495354970917, 0.17833092082704793, 0.0469662038533105, -0.0028577895152307304,
         0.29099850796744287, 0.046137231761455566, -0.20923841559858444, -0.06613100298669164,
         -0.06822448421043388, 0.07665884568148376, 0.31339918388268906, 0.17848512473276362,
         -0.12257719082558292, -0.09916927562478682, -0.07495972834085389, 0.06413205706058327,
         0.1444125551281154, 0.6089459982604638, 0.17463101054296204, 0.05335403311654184,
         -0.012738255533159106, 0.034741076886942744, -0.8123956655755472, -0.04688727359339901,
         0.2019353311723676, 0.20311115159355098, -0.03935654449686459, 0.06967517803561558,
         -0.015536553796198381, -0.034052746766077585, -0.0652802475349671, 0.12250091921016126,
         0.13992004933389163, -0.1744630454565621, -0.08011841031916592, 0.08495219260330641,
         -0.010416448746240264, -0.13704901119491045, 0.20127087898234736, 0.10069294277050474,
         0.006530070028596603, 0.01685149976465394],
    100: [0.05209831906544922, -0.09711438928457942, -0.138075629216013, 0.11075337240412474,
          -0.02722767104792735, -0.003264471482315391, 0.031763779949946995, -0.050768642637863216,
          0.15321644920512753, -0.023673795881357665, -0.007855259602505521, 0.08436049125621169,
          -0.08042013684556613, -0.08836746169393465, -0.01713612684538098, 0.07352562261928337,
          -0.16472255294923277, 0.054735397909665105, 0.15367049429337684, -0.052840216828511186,
          -0.16474394753100705, -0.008949042325014492, -0.13604238879992814, -0.03889380184957428,
          -0.09204607374805457, 0.028746299060313972, 0.024459615449687708, 0.19419303134052282,
          -0.03298012758543287, 0.005092942558421198, 0.014690349519918036, -0.15542921839256926,
          0.03542781035241059, -0.02936262004856886, 0.013728982285007282, -0.060675839686646714,
          0.020253971147053164, -0.14560238458236616, 0.058238549507300195, 0.01729453965350055,
          0.16282166188247535, 0.18634667508905506, -0.0633785151880664, 0.1306728937577658,
          -0.11122706198162254, 0.02721630492005511, 0.0386790859765103, 0.15675607494404736,
          0.013449279190418594, 0.19424679785532378, -0.012188616851492173, 0.03659233047405443,
          -0.0823539082942906, -0.2442046813428865, 0.07523774753713008, 0.4642307845158699,
          0.06318520216483453, 0.0508130532519789, -0.38146942508264287, -0.20739722881342532,
          0.034894137345288796, -0.18234748179875326, 0.09021320290831103, -0.025041033717308155,
          -0.22256541422255027, 0.03383023830474673, -0.13379322735747626, -0.14375439200387638,
          -0.11264177788412304, -0.03744004196587934, 0.06188907762336614, 0.09650583074723643,
          0.08384275663050354, 0.19646375306184494, -0.0744609801686033, 0.009218827950820505,
          0.030343143401107053, -0.02482691343694564, 0.2756343827275015, 0.02422176819772758,
          -0.23416686732715053, -0.05235205832111843, 0.10200689680027618, -0.036736439307184246,
          0.2940298138773312, 0.05685075856419903, 0.017595530040245354, 0.07998299526964742,
          -0.07554345715933386, 0.14788664592348352, 0.016906244366967138, 0.07576796824799294,
          0.07596104636801292, -0.10799951691485174, 0.20830310001378252, -0.07841304845814367,
          0.08663635324040432, 0.12381253234090335, -0.23434524035210175, -0.009255162799313205],
    300: [0.22418554170436922, -0.2888188736390646, 0.138541357137267, 0.0036572450297924063,
          -0.12870587566293681, 0.10243988556804287, 0.061628219635895494, 0.0731760583646325,
          -0.0613547212184861, -1.3476364410168673, 0.42038515953069405, -0.0635953695515016,
          -0.09683484601867541, 0.18085932700844823, 0.23704440249320086, 0.014127583424270562,
          0.17009625902230582, -1.1491660951570601, 0.3149849996143359, 0.06622184613310005,
          0.02468698947696239, 0.07669413923456651, 0.1385161599712863, 0.02130039138707397,
          -0.06640359734343572, -0.010335409190484604, 0.13523394398205724, -0.04214412637655219,
          -0.11938741867849068, 0.0069503700096447745, 0.13332962997155362, -0.18276167442913113,
          0.05238404746251411, 0.008941684533653623, -0.23957182177474648, 0.08500261012699807,
          -0.0068956686792796505, 0.001585649640546265, 0.06339075047010083, 0.19177510530388162,
          -0.13113527555029503, -0.112954309208339, -0.14277442569737858, 0.034139663759527934,
          -0.03427922526924725, -0.0513647302329928, 0.18891632038639494, -0.16673291125863207,
          -0.057783085014844124, 0.036820787536917254, 0.08078611241831335, 0.022950041165501308,
          0.033295092583956534, 0.0117845983464303, 0.05643268887036145, -0.04277245653365069,
          0.011960320790545303, 0.011553850748179662, -0.0007971149632940848, 0.11299947462601066,
          -0.03136808965879448, -0.006155671403386922, -0.009047691481520488, -0.4153417842604509,
          -0.18869843795914445, 0.13708761739174063, 0.005909493593565283, -0.11303380471479721,
          -0.030095582551034127, -0.2390910471525192, -0.0535400792106943, -0.04490346881852842,
          -0.20228351736298425, 0.006565544583374304, -0.09579143714675627, -0.0739203225321337,
          -0.06487372589139846, 0.11173572589203852, -0.04864782293882542, -0.16565203401071177,
          -0.05203598579885941, -0.07896650343295339, 0.13685094309074886, 0.07575291584933341,
          -0.006274910351239313, 0.28693329339684076, 0.5201764953996019, -0.08771337358289512,
          -0.3301086794928012, -0.13596028895608078, 0.11489539130990332, -0.0974435749762466,
          0.06269618995894977, 0.12118785355390706, -0.08026214801168466, 0.35256415142616415,
          -0.06001737527019796, -0.04889804026221007, -0.0682898875211503, 0.08874177983072284,
          0.003964601452043383, -0.07663173974986918, 0.12638968454354807, 0.07809272354381211,
          -0.023162697327602524, -0.5680638179019137, -0.0378934962494425, -0.13509340098553527,
          -0.11351423517925423, -0.11143209971662325, -0.09050222692105529, 0.2517331688607794,
          -0.14841960406948976, 0.034635320617350585, -0.07334758353578258, 0.06319767667118398,
          -0.03834155722904611, -0.05413480907528069, 0.042197910253788426, -0.09038230108266276,
          -0.07052500971990802, -0.009172383100691248, 0.009070340005131338, 0.14051530265403417,
          0.029584354662691846, -0.036431554979162256, -0.08625590921364512, 0.04295038394857042,
          0.08230883498699368, 0.09032880391496495, -0.12279582878558945, -0.013898329297464394,
          0.04811924167840886, 0.08678346997181362, -0.14450626513536827, -0.04425178102100824,
          0.018318914132675074, 0.015028052776074816, -0.1005266209986208, 0.06021413631937442,
          0.7405919638385154, -0.001632564326043843, -0.24960332519721762, -0.02374004719756911,
          0.01639828306796921, 0.11928674727735232, 0.1395087479980048, -0.03162486767249477,
          -0.016447870430264, 0.1407965674541243, -0.0002828534869532372, -0.08052878815607736,
          -0.0021292019541157762, -0.025353495287834087, 0.08693567568203367, 0.1430839274102,
          0.17146175594249208, -0.1394288774913279, 0.048792162778028676, 0.09275186899585167,
          -0.05316838583972354, 0.03110212880228318, 0.012353189134857043, 0.21058272590609764,
          0.3261794020733153, 0.18016021816481465, -0.1588118500841672, 0.15323277898890386,
          -0.22559582202248843, -0.042009798277384214, 0.008469509700446165, 0.03815496958103433,
          0.15187720070327654, 0.13274262589923003, 0.11375661232420824, -0.0952744850987046,
          -0.049488067839782185, -0.10265994359650703, -0.27064332737225794, -0.034566075759255306,
          -0.018812891452488264, -0.0010344250101871578, 0.10340008621077747, 0.13883078128137427,
          0.21130532045778608, -0.019812251504324002, 0.18333542042109632, -0.10751574875949964,
          -0.03128881860577323, 0.025184422764375543, 0.23232949809175724, 0.04205041653514209,
          0.1173164354604317, -0.15506233597871658, 0.006356372922645831, -0.15429841363338256,
          0.1511674343858217, 0.1274574732941446, 0.25768715769514533, -0.25485359970609495,
          -0.07094598190406098, 0.1798345746669871, 0.05402681853390259, -0.09884565840635787,
          -0.2459489245165399, -0.09302717258039374, -0.028205862126332044, 0.09439880511678796,
          0.09234152830648136, 0.029290450928871815, 0.1311056970616741, 0.15682905730417332,
          -0.01691927750214016, 0.23927833216784566, -0.13432887145937242, -0.2242257856940957,
          0.146353196350476, -0.064996993770752, 0.4703618206281071, -0.027190295972943573,
          0.06224710079551949, -0.09135947769946304, 0.21489798102208227, -0.19562410707702874,
          -0.10032276332984857, -0.09056936496143535, -0.06203767777367444, -0.1887666713521054,
          -0.10962988284335924, -0.2773500668329066, 0.1261623902262152, -0.022181522000642523,
          -0.16058579937495204, -0.08047378635817014, 0.02695385511648106, 0.11072885660759921,
          0.014893361073070724, 0.09417023641014764, 0.14300306132727822, -0.1594041133515557,
          -0.06608101785598289, -0.00799541815658455, -0.11668818550953051, -0.1308183365604761,
          -0.09237499304009239, 0.14741082013353243, 0.09179995449548507, 0.08173397944931225,
          0.3211163630073062, -0.0036551715056242634, -0.047030574233688824, -0.02311823191269146,
          0.04896041301279729, 0.08669897747107991, -0.06766318347501796, -0.5002850004994835,
          -0.04851682061682701, 0.14144484177034675, -0.03299551018921661, -0.11954620115014937,
          -0.14929710110787, -0.2388326199010263, -0.01988600809216346, -0.1591715914752679,
          -0.05208224610011773, 0.28009816930464504, -0.002911717009782663, -0.054579906742433494,
          -0.4738466477288287, 0.17112231903532513, -0.12066773401448566, -0.04217406117091595,
          0.13953421602242932, 0.26114828443918686, 0.012869335316309611, 0.009291544772323388,
          -0.0026464363047812058, -0.07533078170966552, 0.017842164966117263, -0.26869538091901846,
          -0.218193932593655, -0.170851350359631, -0.10228003018196515, -0.055290122046996966,
          0.13514348272968923, 0.12362528635977957, -0.10980561644954541, 0.13979893806750074,
          -0.20233739314044072, 0.08813353931898646, 0.38496243831577515, -0.10653732679592942,
          -0.061994460509923556, 0.02885062873759481, 0.03230145614980359, 0.02385628066949197,
          0.06994879464652057, 0.19310566556945175, -0.07767794356709506, -0.14481313231293244]
}


def read_embeddings(file_enc, skip_lines=0, filter_set=None, max_words=None):
    embs = dict()
    total_vectors_in_file = 0
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            total_vectors_in_file += 1
            if filter_set is not None and l_split[0] not in filter_set:
                continue
            embs[l_split[0]] = np.array([float(em) for em in l_split[1:]])
            if max_words and len(embs) >= max_words:
                break
    return embs, total_vectors_in_file


class GloveIndex:
    def __init__(self, mapping, dims: int):
        self.mapping = mapping
        self.unk_vec = np.array(AVG_ALL_VECS[dims])
        self.pad_vec = np.zeros((dims,))
        self.dims = dims

    def get_vec(self, word: str, always_return_vec: bool = False) -> Optional[np.array]:
        """
        Converts a word to a vector. If the word is not in the vocab then it
        will return None. However, if always_return_vec is true, it will
        correctly handle UNK and PAD and return a vector for them
        """
        if word == parse_constants.PAD:
            return self.pad_vec
        return self.mapping.get(word, self.unk_vec if always_return_vec else None)

    def __contains__(self, word: str):
        return word in self.mapping

    def __index__(self, word: str) -> List[float]:
        return self.get_vec(word)


def get_glove_words(dimensionality: int, vocab: Vocab) -> GloveIndex:
    glove_file = _get_file(dimensionality)
    word_set = set(iter(vocab))
    print(word_set)
    vec_mapping, _ = read_embeddings(glove_file, filter_set=word_set)
    print(f"glove found {len(vec_mapping)} words")

    return GloveIndex(vec_mapping, dimensionality)


def _get_file(dims) -> str:
    dirname = os.path.dirname(os.path.abspath(__file__))
    glove_file = os.path.join(dirname, GLOVE_DIR, GLOVE_DIM_TO_NAME[dims])
    return glove_file


def _get_average_of_all_vecs(dims: int):
    glove_file = _get_file(dims)
    seen = 0
    sums = np.zeros(dims)
    with open(glove_file, 'rb') as f:
        for i, line in enumerate(f):
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            line = line.decode('utf8')
            l_split = line.strip().split(' ')
            if len(l_split) == 2:
                continue
            seen += 1
            try:
                sums += np.array([float(em) for em in l_split[1:]])
            except ValueError as e:
                print(line)
    return list(sums / seen)


def _get_word_list(dims: int):
    glove_file = _get_file(dims)
    words = []
    with open(glove_file, 'rb') as f:
        for i, line in enumerate(f):
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            words.append(l_split[0])
    return words


if __name__ == "__main__":
    print(_get_average_of_all_vecs(100))
    #words = _get_word_list(50)
    #with open("glove_vocab.txt", "w") as f:
    #    f.writelines("\n".join(words))
