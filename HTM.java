package org.numenta.nupic.examples.qt;

import org.numenta.nupic.Parameters;
import org.numenta.nupic.algorithms.BPClassifier;
import org.numenta.nupic.algorithms.SpatialPooler;
import org.numenta.nupic.algorithms.TemporalMemory;
import org.numenta.nupic.encoders.ScalarEncoder;
import org.numenta.nupic.model.Cell;
import org.numenta.nupic.model.ComputeCycle;
import org.numenta.nupic.model.Connections;
import org.numenta.nupic.util.ArrayUtils;
import org.numenta.nupic.util.FastRandom;

import java.io.*;
import java.util.*;


public class HTM {

    static boolean isResetting = true;

    public static void main(String[] args) throws FileNotFoundException {
        // 1. 初始化参数配置
        Parameters parameters = getParameters();
        isResetting = true; // 标记是否需要重置时间记忆状态
        int inputSize = 2160;
        //int hiddenSize = 1080;
        int hiddenSize = 360;
        int outputSize = 360;
        double learningRate = 0.05;

        //Layer components
        // 2. 初始化编码器、空间池、时间记忆和分类器
        ScalarEncoder.Builder EncodeBuilder =
                ScalarEncoder.builder()
                        .n(360)//代表输出编码的总位数
                        .w(7)//数字编码位的数量
                        .minVal(0)//输入信号的最小值
                        .maxVal(360)//输入信号的最大值
                        .forced(true);//是否进行输入合法性检查
        ScalarEncoder encoder = EncodeBuilder.build();

        SpatialPooler spatialPooler = new SpatialPooler();  // 空间池：稀疏化输入
        TemporalMemory temporalMemory = new TemporalMemory();   // 时间记忆：处理时序依赖
        /*SDRClassifier classifier = new SDRClassifier(new TIntArrayList(new int[] { 0 }),// 预测步长（1步）
                0.1, // 学习率（alpha）  高的学习率可能导致模型快速适应新数据，但不稳定；低的学习率则可能导致学习缓慢，但更稳定
                0.3, // 实际值平滑因子（actValueAlpha）  新值的权重是30%，旧值的权重是70%。这有助于减少噪声的影响，使预测更平滑。
                2); // 调试信息级别（verbosity）    表示输出较多的调试信息
        */
        // 3. 构建网络层（整合所有组件）
        
        BPClassifier classifier = new BPClassifier(inputSize, hiddenSize, outputSize, learningRate);
        
        Layer<int[]> layer = getLayer(parameters, encoder, spatialPooler, temporalMemory, classifier);

        //读取文件
        // 4. 读取训练数据
        //String FilePath = "D:\\workspace\\adapt_RAN\\array_datas-every-third-line.txt";
        //String FilePath_angle = "D:\\workspace\\adapt_RAN\\angle_datas_every_third_line.txt";
        String FilePath = "D:\\workspace\\adapt_RAN\\array_datasf.txt";
        String FilePath_angle = "D:\\workspace\\adapt_RAN\\angle_datasf.txt";
        List<Double> angleNumbers = new ArrayList<>(); //存储一周角度数组
        List<int []> synsNumbers = new ArrayList<>(); //存储一周神经元数组

        try {
            Scanner scannerArray = new Scanner(new FileReader(FilePath));
            Scanner scannerAngle = new Scanner(new FileReader(FilePath_angle));
            while (scannerArray.hasNextLine()){
                String[] line = scannerArray.nextLine().trim().split(" ");//读一行
                int[] synNumbers = new int[line.length]; //存储神经元数组

                for (int i = 0; i < line.length; i++) {
                    synNumbers[i] = Integer.parseInt(line[i]);
                }

                synsNumbers.add(synNumbers);
            }
            while (scannerAngle.hasNextLine()){
                String[] line = scannerAngle.nextLine().trim().split(" ");//读一行
                double angleNumber = -1.0;
                /*for (int i = 0; i < line.length; i++) {
                    angleNumber = Double.parseDouble(line[i]);
                }*/
                angleNumber = Double.parseDouble(line[0]);
                if (angleNumber > -1){
                    angleNumbers.add(angleNumber);
                }

            }

        }catch (IOException e){
            System.out.println("读取文件错误: "+ e.getMessage());
        }

        /* i为成功遍历数据的迭代轮数；x为迭代的总次数;j为每一轮循环迭代  */
        //i为总学习次数   x为总共迭代次数   j恒为0可以用0代替
        for(int i = 0, x = 0 ; i < 400 ; i++, x += 360) {
            if (i == 0 && isResetting) {
                System.out.println("reset:");
                temporalMemory.reset(layer.getMemory());
            }

            runThroughLayer(layer, synsNumbers, angleNumbers, 0, x);
        }
    }


    public static Parameters getParameters() {
        Parameters parameters = Parameters.getAllDefaultParameters();
        // --- 空间池（Spatial Pooler）参数 ---
        parameters.set(Parameters.KEY.CELLS_PER_COLUMN, 6); // 每列包含6个细胞（时间记忆模块的单元）
        // 输入层配置
        parameters.set(Parameters.KEY.INPUT_DIMENSIONS, new int[]{360});       // 输入维度（需与编码器输出位数一致）
        parameters.set(Parameters.KEY.COLUMN_DIMENSIONS, new int[]{360});      // 列的数量（空间池输出稀疏表示的维度）

        // 潜在连接配置
        parameters.set(Parameters.KEY.POTENTIAL_RADIUS, new int[]{10});        // 潜在连接半径（列可连接输入的最大距离，基于拓扑）
        parameters.set(Parameters.KEY.POTENTIAL_PCT, 1.0);                    // 潜在连接比例（1.0表示半径内所有输入都可能连接）

        // 抑制机制
        parameters.set(Parameters.KEY.GLOBAL_INHIBITION, true);               // 全局抑制（选择全局最活跃的列，而非局部区域）
        parameters.set(Parameters.KEY.LOCAL_AREA_DENSITY, -1.0);              // 局部密度（-1表示禁用，改用NUM_ACTIVE_COLUMNS_PER_INH_AREA）
        parameters.set(Parameters.KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 48.0); // 每个抑制区域激活的列数（控制稀疏性）

        // 突触激活与学习规则
        parameters.set(Parameters.KEY.STIMULUS_THRESHOLD, 3.0);               // 列激活的最小激活突触数（过滤噪声）
        parameters.set(Parameters.KEY.SYN_PERM_INACTIVE_DEC, 0.0005);         // 非激活突触的持久度衰减量（弱化无关连接）
        parameters.set(Parameters.KEY.SYN_PERM_ACTIVE_INC, 0.0015);           // 激活突触的持久度增量（强化相关连接）
        parameters.set(Parameters.KEY.SYN_PERM_TRIM_THRESHOLD, 0.05);         // 突触修剪阈值（低于此值的突触被移除，保持稀疏性）
        parameters.set(Parameters.KEY.SYN_PERM_CONNECTED, 0.1);               // 突触连接阈值（持久度≥此值视为“已连接”）

        // 占空周期与Boosting机制
        parameters.set(Parameters.KEY.MIN_PCT_OVERLAP_DUTY_CYCLES, 0.1);      // 最小重叠占空比（防止列长期不激活）
        parameters.set(Parameters.KEY.MIN_PCT_ACTIVE_DUTY_CYCLES, 0.1);       // 最小活跃占空比（强制列的最低激活频率）
        parameters.set(Parameters.KEY.DUTY_CYCLE_PERIOD, 10);                 // 占空周期计算的时间窗口（步数）
        parameters.set(Parameters.KEY.MAX_BOOST, 10.0);                       // Boosting的最大增益倍数（增强不活跃列的竞争力）

        // 随机性与实验复现
        parameters.set(Parameters.KEY.SEED, 42);                              // 随机种子（确保实验可重复）
        parameters.set(Parameters.KEY.RANDOM, new FastRandom());              // 随机数生成器（优化计算速度）

        // --- 时间记忆（Temporal Memory）参数 ---
        parameters.set(Parameters.KEY.INITIAL_PERMANENCE, 0.2);               // 新突触的初始持久度（初始连接强度）
        parameters.set(Parameters.KEY.CONNECTED_PERMANENCE, 0.7);             // 突触被视为“已连接”的持久度阈值
        parameters.set(Parameters.KEY.MIN_THRESHOLD, 4);                      // 树突段激活的最小突触数（过滤噪声）
        parameters.set(Parameters.KEY.MAX_NEW_SYNAPSE_COUNT, 6);              // 每次学习新增突触的最大数量（控制复杂度）
        parameters.set(Parameters.KEY.PERMANENCE_INCREMENT, 0.1);             // 正确预测时突触持久度的增量（强化学习）
        parameters.set(Parameters.KEY.PERMANENCE_DECREMENT, 0.1);             // 错误预测时突触持久度的减量（弱化噪声）
        parameters.set(Parameters.KEY.ACTIVATION_THRESHOLD, 4);               // 树突段被激活的最小连接突触数（与MIN_THRESHOLD协同）
        parameters.set(Parameters.KEY.LEARNING_RADIUS, 20);                   // 学习邻域半径（影响新突触的连接范围）

        // --- 参数覆盖（调优示例）---
        parameters.set(Parameters.KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 100.0); // 增加活跃列数（提升模型容量）
        parameters.set(Parameters.KEY.DUTY_CYCLE_PERIOD, 100);                // 延长占空周期（平滑长时段统计）


        return parameters;
    }

    public static void runThroughLayer(Layer<int[]> layer, List<int[]> synsNumbers, List<Double> anglesNumbers, int recordNum, int sequenceNum) {
        layer.input(synsNumbers, anglesNumbers, recordNum, sequenceNum);
    }

    public static Layer<int[]> getLayer(Parameters p, ScalarEncoder e, SpatialPooler s, TemporalMemory t, BPClassifier c) {
        Layer<int[]> layer = new LayerImpl(p, e, s, t, c);
        return layer;
    }

    public interface Layer<T> {
        void input(List<int[]> binaryDatas, List<Double> anglesNumbers, int recordNum,  int sequenceNum);
        int[] getPredicted();
        Connections getMemory();
        int[] getActual();
    }

    public static class LayerImpl implements Layer<int[]> {
        private Parameters params;
        private Connections memory = new Connections();
        private Map<String, Object> classification = new LinkedHashMap<String, Object>();
        private ScalarEncoder encoder;
        private SpatialPooler spatialPooler;
        private TemporalMemory temporalMemory;
        private BPClassifier classifier;
        private int columnCount;
        private int cellsPerColumn;
        private int theNum;//标记迭代轮数
        private int[] actual;
        private int[] predictedColumns;
        private int[] lastPredicted;

        public LayerImpl(Parameters p, ScalarEncoder e, SpatialPooler s, TemporalMemory t, BPClassifier c) {
            this.params = p;
            this.encoder = e;
            this.spatialPooler = s;
            this.temporalMemory = t;
            this.classifier = c;
            params.apply(memory);
            spatialPooler.init(memory);
            TemporalMemory.init(memory);

            columnCount = memory.getPotentialPools().getMaxIndex() + 1;
            cellsPerColumn = memory.getCellsPerColumn();
        }
        
        /**
         * 把指定的索引数组转换为指定长度的向量，该向量原则是只包含0和1元素的向量
         * @param indexs 指定的索引列表
         * @param vectLength
         * @return
         */
        public double[] convetIndexToBinaryVect(int[] indexs,int vectLength)
        {
        	 // 将激活位置的索引转成1，其余为0
        	double[] binaryVector = new double[vectLength];
            for (int j = 0; j < vectLength; j++) {
                if (Arrays.binarySearch(indexs, j) >= 0) 
                {
                	binaryVector[j]=1d;
                } else {
                	binaryVector[j]=0d;
                }
            }
            return binaryVector;
        }
        
        

        /**
         * 找到double数组中最大值元素的索引
         * @param array
         * @return
         */
        public static int findMaxIndex(double[] array) 
        {
            if (array == null || array.length == 0) {
                throw new IllegalArgumentException("数组不能为空");
            }

            int maxIndex = 0;
            double maxValue = array[0];

            for (int i = 1; i < array.length; i++) {
                if (array[i] > maxValue) {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }
        public void input(List<int[]> synsNumbers, List<Double> anglesNumbers, int recordNum,  int sequenceNum) {
            theNum++;
            try {
                //recordNum 本轮循环的次数    sequenceNum 循环的总次数   theNum 迭代次数
                for(int[] synNumbers0 : synsNumbers) {
                    System.out.println();
                    System.out.println("===== " + " 每一轮迭代循环中的次数--Record Num: " + recordNum + " =====" + " 循环迭代总次数--Sequence Num: " + sequenceNum + " =====");

                    int[] combinedEncodedNums0 = synNumbers0;

                    System.out.println("神经元状态编码后的值为：" + Arrays.toString(combinedEncodedNums0));

                    int[] output0 = new int[columnCount];//记录每个列的输出值的数组
                    spatialPooler.compute(memory, combinedEncodedNums0, output0, true); //处理输入数据并生成活跃列的索引
                    System.out.println("SpatialPooler Output = " + Arrays.toString(output0));

                    recordNum++;//每一轮迭代循环中的次数
                    sequenceNum++;//循环迭代总次数
                }

                //首先让SpatialPooler单独训练（预热），迭代400轮以后再去训练TemporalMemory

                if (theNum == 400) {
                    int sequenceNum0 = 0;
                    for (int itertimes = 0; itertimes < 201; itertimes++) {
                        int index = 0;
                        for (int[] synNumbers : synsNumbers) {
                            System.out.println();
                            System.out.println("===== " + " 每一轮迭代循环中的次数--itertimes Num: " + itertimes + " =====" + " 循环迭代总次数--Sequence Num: " + sequenceNum + sequenceNum0 + " =====");

                            double angle = anglesNumbers.get(index);
                            int[] combinedEncodedNums = synNumbers;

                            int[] output = new int[columnCount];//记录每个列的输出值的数组
                            spatialPooler.compute(memory, combinedEncodedNums, output, true);   //处理输入数据并生成活跃列的索引
                            System.out.println("SpatialPooler Output = " + Arrays.toString(output));

                            System.out.println("-------开始tm-------");

                            int[] input_predict = actual = ArrayUtils.where(output, ArrayUtils.WHERE_1);//SpatialPool轮实际激活的列的索引 找到 `output` 数组中所有值为1的索引位置
                            ComputeCycle cc_easy = temporalMemory.compute(memory, input_predict, true);
                            lastPredicted = predictedColumns;
                            predictedColumns = getSDR(cc_easy.predictiveCells());//Get the predicted column indexes获取了预测单元集合所对应的列的索引集合
                            int[] activeCellIndexes_easy = Connections.asCellIndexes(cc_easy.activeCells()).stream().mapToInt(p -> p).sorted().toArray();  // 获取当前时间步中所有激活的细胞索引，供分类器使用//Get the active cells for classifier input

                            int bucketIdx = (int) angle;                      
                            System.out.println("  |  胜出的神经元 = " + Arrays.toString(activeCellIndexes_easy));

                            ///这里需要把activeCellIndexes_easy转换成为完全的输入向量
                            double[] inputVect=convetIndexToBinaryVect(activeCellIndexes_easy, 2160);
                            int[] targetIndexes=new int[]{bucketIdx};
                            double[] targetOutputVect=convetIndexToBinaryVect(targetIndexes, 360);
                            
                            //classifier.setPeriodic(true);
                            double[] result = classifier.compute(sequenceNum0,inputVect,targetOutputVect,true,true);

                            int synNumbers_pre = findMaxIndex(result);

                            System.out.println("  |  实际角度角度 = " + angle + "\n");
                            System.out.println("  |  CLAClassifier 预测角度 = " + synNumbers_pre + "\n");

                            sequenceNum0++;
                            if(index < synsNumbers.size())
                                index++;

//                            pw2.print(angle + " ");
//                            pw3.print(synNumbers_pre + " ");
                            double error = synNumbers_pre - angle;
                            if (error >= 180) {
                                error = 360 - error;
                            } else if (error <= -180) {
                                error = 360 + error;
                            }

//                            pw1.print(error + " ");
                        }
                    }
                    String w_path1 = "D:\\workspace\\adapt_RAN\\data_err_360.txt";
                    String w_path2 = "D:\\workspace\\adapt_RAN\\data_rea_360.txt";
                    String w_path3 = "D:\\workspace\\adapt_RAN\\data_pre_360.txt";
                    PrintWriter pw1 = new PrintWriter(new FileWriter(w_path1,true));
                    PrintWriter pw2 = new PrintWriter(new FileWriter(w_path2,true));
                    PrintWriter pw3 = new PrintWriter(new FileWriter(w_path3,true));

                    //读取文件
                    //String FilePath = "D:\\workspace\\adapt_RAN\\array_test.txt";
                    //String FilePath_angle = "D:\\workspace\\adapt_RAN\\angle_test.txt";
                    
                    //读取文件
                    String FilePath = "D:\\workspace\\adapt_RAN\\array_datast.txt";
                    String FilePath_angle = "D:\\workspace\\adapt_RAN\\angle_datast.txt";
                    List<Double> angleNumber_test = new ArrayList<>(); //存储一周角度数组
                    List<int[]> synsNumbers_test = new ArrayList<>(); //存储一周神经元数组

                    try {

                        Scanner scannerArray = new Scanner(new FileReader(FilePath));
                        Scanner scannerAngle = new Scanner(new FileReader(FilePath_angle));
                        while (scannerArray.hasNextLine()) {
                            String[] line = scannerArray.nextLine().trim().split(" ");//读一行
                            int[] synNumbers = new int[line.length]; //存储神经元数组

                            for (int i = 0; i < line.length; i++) {
                                synNumbers[i] = Integer.parseInt(line[i]);
                            }

                            synsNumbers_test.add(synNumbers);
                        }
                        while (scannerAngle.hasNextLine()) {
                            String[] line = scannerAngle.nextLine().trim().split(" ");//读一行
                            double angleNumber = -1.0;
                    /*for (int i = 0; i < line.length; i++) {
                        angleNumber = Double.parseDouble(line[i]);
                    }*/
                            angleNumber = Double.parseDouble(line[0]);
                            if (angleNumber > -1) {
                                angleNumber_test.add(angleNumber);
                            }

                        }

                    } catch (IOException e) {
                        System.out.println("读取文件错误: " + e.getMessage());
                    }


//              pw.println("-----------------开始预测------------------");
                    int sequenceNum1 = 0;
                    int index1 = 0;
                    for (int[] synNumbers1 : synsNumbers_test) {
                        double angle1 = angleNumber_test.get(index1);
                        System.out.println();
                        int[] combinedEncodedNums1 = synNumbers1;

                        System.out.println("神经元状态编码后的值为：" + Arrays.toString(combinedEncodedNums1));

                        int[] output1 = new int[columnCount];//记录每个列的输出值的数组
                        spatialPooler.compute(memory, combinedEncodedNums1, output1, false);
                        System.out.println("SpatialPooler Output = " + Arrays.toString(output1));

                        System.out.println("-------开始tm-------");

                        int[] input_predict1 = actual = ArrayUtils.where(output1, ArrayUtils.WHERE_1);//SpatialPool轮实际激活的列的索引
                        ComputeCycle cc_easy1 = temporalMemory.compute(memory, input_predict1, false);
//                        lastPredicted = predictedColumns;
                        predictedColumns = getSDR(cc_easy1.predictiveCells());//Get the predicted column indexes 获取了预测单元集合所对应的列的索引集合
                        int[] activeCellIndexes_easy1 = Connections.asCellIndexes(cc_easy1.activeCells()).stream().mapToInt(p -> p).sorted().toArray();  //Get the active cells for classifier input
                        System.out.println("TemporalMemory Input = " + Arrays.toString(input_predict1));
                        System.out.println("TemporalMemory Prediction = " + Arrays.toString(predictedColumns));

                        System.out.println("  |  胜出的神经元 = " + Arrays.toString(activeCellIndexes_easy1));

                        
                        ///这里需要把activeCellIndexes_easy转换成为完全的输入向量
                        double[] inputVect1=convetIndexToBinaryVect(activeCellIndexes_easy1, 2160);
                        //int[] targetIndexes1=new int[]{bucketIdx1};
                        //double[] targetOutputVect1=null;
                        
                        //classifier.setPeriodic(true);
                        double[] result1 = classifier.compute(sequenceNum1,inputVect1,null,false,true);

                        int synNumbers_pre1 = findMaxIndex(result1);


                        System.out.println("  |  实际角度 = " + angle1 + "\n");
                        System.out.println("  |  CLAClassifier 预测角度 = " + synNumbers_pre1 + "\n");

                        pw2.print(angle1 + " ");
                        pw3.print(synNumbers_pre1 + " ");
                        double error = synNumbers_pre1 - angle1;
                        if (error >= 180) {
                            error = 360 - error;
                        } else if (error <= -180) {
                            error = 360 + error;
                        }

                        pw1.print(error + " ");
                        index1++;
                    }


                    pw1.close();
                    pw2.close();
                    pw3.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private int[] combineListENcodedNums(List<int[]> encoderNumS) {
            int combinedLength = encoderNumS.size() * encoderNumS.get(0).length;     //初始化数组大小
            int[] combinedEncodedNumbers = new int[combinedLength]; //初始化数组

            int idx = 0;
            for (int[] encodedNumber : encoderNumS) {    //双层循环存储数组
                for (int value : encodedNumber) {
                    combinedEncodedNumbers[idx++] = value;
                }
            }
            return combinedEncodedNumbers;
        }

        public int[] getPredicted() {
            return lastPredicted;
        }

        public Connections getMemory() {
            return memory;
        }

        public int[] getActual() {
            return actual;
        }

        public int[] getSDR(Set<Cell> cells) {
            int[] retVal = new int[cells.size()];
            int i = 0;
            for (Cell cell : cells) {
                retVal[i] = cell.getIndex() / cellsPerColumn;
                i++;
            }
            Arrays.sort(retVal);
            retVal = ArrayUtils.unique(retVal);

            return retVal;
        }
    }
}
