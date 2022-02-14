package instance;

import lombok.AllArgsConstructor;
import lombok.Getter;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

@Getter
@AllArgsConstructor
public class InstanceManager {
    private Instances train;
    private Instances test;
    private Instances all;
    private int classIndex;
    private double percentage;
    private String fileName;
    private static final String PATH =  "src/main/resources/files/";
    private static final String ARFF_SUFFIX = ".arff";

    public InstanceManager() {
        classIndex = -1;
        percentage = -1d;
        fileName = "heart";
        process();

    }

    public InstanceManager(String p_fileName) {
        classIndex = -1;
        percentage = -1d;
        fileName = p_fileName;
        process();

    }

    public InstanceManager(int classIndex, double percentage) {
        this.classIndex = classIndex;
        this.percentage = percentage;
        fileName = "heart";
        process();
    }


    private void process()  {
        DataSource source;
        try {
            String help = PATH + fileName + ARFF_SUFFIX;
            source = new DataSource(help);
            all = source.getDataSet();
        } catch (Exception e ) {
            System.err.println("Exception on " + e);
            throw new RuntimeException();
        }

        System.out.println(all.toSummaryString());

        classIndex = classIndex == -1 ? all.numAttributes() - 1 : classIndex;

        all.setClassIndex(classIndex);

        //randomize order of records in my input dataset
        //all.randomize(new Random(23));

        //how much data in dataset, e.g. 80 percent
        percentage = percentage == -1d ? 0.80d : percentage;

        int trainSize = (int) Math.round(all.numInstances() * percentage);
        int testSize = all.numInstances() - trainSize;

        //from 0 to train size are train instances
        train = new Instances(all, 0, trainSize);

        //from train size to test size are test instances
        test = new Instances(all, trainSize, testSize);
    }


    public Instances getNewInstance(double[] data) {
        Instances newInstances = new Instances(
                "Target",
                getALlAttributes(),
                2
        );
        newInstances.add(new DenseInstance(1d, data));
        return newInstances;
    }

    private ArrayList<Attribute> getALlAttributes() {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < all.numAttributes(); i++) {
            attr.add(all.firstInstance().attribute(i));
        }
        return attr;
    }

    public void printInstances() {
        System.out.println(all.toString());
    }
}
