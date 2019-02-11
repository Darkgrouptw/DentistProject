#include "OBJLoader.h"

// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide : 
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc
bool OBJLoader::loadOBJ(
	QString path,
	QVector<QVector3D> & out_vertices,
	QVector<QVector2D> & out_uvs,
	QVector<QVector3D> & out_normals,
	QVector<unsigned int> &out_materialIndices,
	QVector<QString> &out_mtls)
{
	//qDebug(qPrintable("Loading OBJ file " + path + "..."));
	QVector<unsigned int> vertexIndices, uvIndices, normalIndices, materialIndices;
	QVector<QVector3D> temp_vertices;
	QVector<QVector2D> temp_uvs;
	QVector<QVector3D> temp_normals;
	QVector<QString> temp_mtls;

	QFile file(path);
	if (!file.open(QIODevice::ReadOnly)){
		qDebug(qPrintable("Impossible to open the .obj file ! Are you in the right path ?"));
		return false;
	}
	int fcount = 0;//face count for material

	QTextStream in(&file);
	float x, y, z;
	while (1){
		QString lineHeader;

		in >> lineHeader;
		if (lineHeader.isNull())
			break;

		if (lineHeader == "v")				//'v':°OÂI
		{
			in >> x >> y >> z;
			QVector3D vertex(x, y, z);
			temp_vertices.push_back(vertex);
		}
		else if (lineHeader == "vt")
		{
			// Invert V coordinate since we will only use DDS texture,
			//which are inverted.Remove if you want to use TGA or BMP loaders.
			//uv.y = -uv.y; 

			in >> x >> y;
			QVector2D uv(x, 1 - y);
			temp_uvs.push_back(uv);
		}
		else if (lineHeader == "vn")
		{
			in >> x >> y >> z;
			QVector3D normal(x, y, z);
			temp_normals.push_back(normal);
		}
		else if (lineHeader == "f")
		{
			//vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			char temp;
			in >> vertexIndex[0] >> temp >> uvIndex[0] >> temp >> normalIndex[0] >>
				vertexIndex[1] >> temp >> uvIndex[1] >> temp >> normalIndex[1] >>
				vertexIndex[2] >> temp >> uvIndex[2] >> temp >> normalIndex[2];
			fcount++;
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices.push_back(uvIndex[0]);
			uvIndices.push_back(uvIndex[1]);
			uvIndices.push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}
		else if (lineHeader == "usemtl")
		{
			out_materialIndices.push_back(fcount);			//last material used face counts
			QString material;

			in >> material;
			out_mtls.push_back(material);
			fcount = 0;//face count for next material
		}
		else
		{
			// Probably a comment, eat up the rest of the line
			QString stupidBuffer;
			stupidBuffer = in.readLine();
		}
	}
	out_materialIndices.push_back(fcount);					//at the end , push the last faces

	// For each vertex of each triangle
	for (unsigned int i = 0; i < vertexIndices.size(); i++){
		// Get the indices of its attributes
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];

		// Get the attributes thanks to the index
		QVector3D vertex = temp_vertices[vertexIndex - 1];
		QVector2D uv = temp_uvs[uvIndex - 1];
		QVector3D normal = temp_normals[normalIndex - 1];

		// Put the attributes in buffers
		out_vertices.push_back(vertex);
		out_uvs.push_back(uv);
		out_normals.push_back(normal);
	}
	return true;
}

bool OBJLoader::loadMTL(
	QString path,
	QVector<QVector3D> &Kd,
	QVector<QVector3D> &Ka,
	QVector<QVector3D> &Ks,
	QVector<QString> &out_name,
	QString &texture)
{
	QFile file(path);
	
	if (!file.open(QIODevice::ReadOnly))
	{
		qDebug(qPrintable("Impossible to open the .mtl file ! Are you in the right path ?"));
		return false;
	}

	QTextStream in(&file);
	float x, y, z;
	while (1)
	{
		QString lineHeader;
		// read the first word of the line
		in >> lineHeader;
		if (lineHeader.isNull())
			break;

		// else : parse lineHeader
		if (lineHeader == "newmtl")
		{
			QString material;
			in >> material;
			out_name.push_back(material);
		}
		else if (lineHeader == "Kd")
		{
			in >> x >> y >> z;
			QVector3D diffuse(x, y, z);
			Kd.push_back(diffuse);
		}
		else if (lineHeader == "Ka")
		{
			in >> x >> y >> z;
			QVector3D ambient(x, y, z);
			Ka.push_back(ambient);
		}
		else if (lineHeader == "Ks")
		{
			in >> x >> y >> z;
			QVector3D specular(x, y, z);
			Ks.push_back(specular);
		}
		else if (lineHeader == "map_Kd")
		{
			in >> texture;
		}
		else
		{
			// Probably a comment, eat up the rest of the line
			QString stupidBuffer;
			stupidBuffer = in.readLine();
		}
	}
	return true;
}