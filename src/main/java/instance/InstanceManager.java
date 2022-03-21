package instance;

import lombok.AllArgsConstructor;
import lombok.Getter;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;


@Getter
@AllArgsConstructor
public class InstanceManager {
    private Instances train;
    private Instances test;
    private Instances all;
    private int classIndex;
    private double percentage;
    private String fileName;
    private static final String PATH = "src/main/resources/files/";
    private static final String ARFF_SUFFIX = ".arff";

    public InstanceManager(String pFileName) {
        classIndex = -1;
        percentage = -1d;
        fileName = pFileName;
        process();

    }

    public InstanceManager(String pFileName, int classIndex) {
        this.classIndex = classIndex;
        fileName = pFileName;
        percentage = -1d;
        process();
    }


    public InstanceManager(String pFileName, int classIndex, double percentage) {
        this.classIndex = classIndex;
        this.percentage = percentage;
        fileName = pFileName;
        process();
    }


    private void process() {
        DataSource source;
        String filePath = "";
        try {
            filePath = PATH + fileName + ARFF_SUFFIX;
            source = new DataSource(filePath);
            all = source.getDataSet();
        } catch (Exception e) {
            throw new RuntimeException("File was not found " + filePath);
        }

        classIndex = classIndex == -1 ? all.numAttributes() - 1 : classIndex;

        all.setClassIndex(classIndex);

        percentage = percentage == -1d ? 0.80d : percentage;

        int trainSize = (int) Math.round(all.numInstances() * percentage);
        int testSize = all.numInstances() - trainSize;

        //from 0 to train size are train instances
        train = new Instances(all, 0, trainSize);

        //from train size to test size are test instances
        test = new Instances(all, trainSize, testSize);
    }
}
