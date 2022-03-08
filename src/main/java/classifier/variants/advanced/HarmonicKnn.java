package classifier.variants.advanced;

import classifier.variants.Variant;
import lombok.Getter;
import lombok.Setter;
import structure.MathOperation;
import structure.Structure;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class HarmonicKnn implements Variant {
    private Structure structure;
    private double[] distances;
    Map<Double, Double> info = new HashMap<>();
    private int m;
    private int k;

    public HarmonicKnn(Structure structure, int k) {
        this.structure = structure;
        this.k = k;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        return HmdKnnAlg(instance, m_NumClasses);
    }


    @Override
    public double classifyInstance(Instance target, int m_NumClasses) {
        double min = Double.MAX_VALUE, endClass = 0d;
        double[] meanDistances = HmdKnnAlg(target, m_NumClasses);
        for (int i = 0; i < meanDistances.length; i++) {
            if (meanDistances[i] < min) {
                min = meanDistances[i];
                endClass = i;
            }
        }
        return endClass;
    }

    @Override
    public String getOption() {
        return "-H";
    }

    private double[] HmdKnnAlg(Instance target, int m_NumClasses) {
        this.m = m_NumClasses;
        Instances neighbours = structure.findKNearestNeighbours(target, k);
        this.distances = structure.getDistances(); //1
        Instances meanInstances = getMeanInstances(neighbours); //2
        double[] harmonicMeanDistances = getHarmonicMeanDistances(target, meanInstances); //3 @TODO AKA SUMA, KDE SUMA
        return getResult(harmonicMeanDistances); //4 @TODO AKA SUMA, KDE SUMA
    }

    private double[] getResult(double[] harmonicMeanDistances) {
        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            result[i] = k / (1 / harmonicMeanDistances[i]);
        }
        return result;
    }

    private double[] getHarmonicMeanDistances(Instance target, Instances meanVectors) {
        double[] harmonicMeanDistances = new double[m];
        double[] denominator = new double[m];
        for (int i = 0; i < m; i++) {
            double distance = MathOperation.euclidDistance(target.classIndex(), target, meanVectors.get(i));
            denominator[i] = 1 / distance;
            harmonicMeanDistances[i] = m / denominator[i];
        }
        return harmonicMeanDistances;
    }

    private Instances getMeanInstances(Instances neighbours) {
        Instances inst = new Instances("target", getALlAttributes(neighbours.firstInstance()), m);
        double[][] meanVectors = new double[m][neighbours.numAttributes()];
        for (int i = 0; i < neighbours.size(); i++) {
            for (int j = 0; j < neighbours.numAttributes(); j++) {
                meanVectors[(int) neighbours.get(i).classValue()][j] += neighbours.get(i).value(j);
            }
        }
        for (int i = 0; i < meanVectors.length; i++) {
            for (int j = 0; j < neighbours.numAttributes(); j++) {
                meanVectors[i][j] = (1 / (double) k) * meanVectors[i][j];
            }
            inst.add(new DenseInstance(1d, meanVectors[i]));
        }
        return inst;
    }


    private ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }

    private void printInfo() {
        System.out.println("\n--------------------HMD-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
    }
}
