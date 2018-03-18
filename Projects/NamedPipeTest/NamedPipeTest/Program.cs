using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.Pipes;

namespace NamedPipeTest
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            byte[] buff = new byte[1024];
            NamedPipeServerStream pipeServer = new NamedPipeServerStream("myPipe", PipeDirection.InOut);
            Console.WriteLine("Waiting for client connection");
            pipeServer.WaitForConnection();
            Console.WriteLine("Client Just connected!");
            StreamReader sr = new StreamReader(pipeServer);
            //Console.WriteLine(sr.ReadLine());
            Encoding unicode = Encoding.Unicode;
            Encoding ascii = Encoding.ASCII;
            string unicodeString = sr.ReadLine();
            byte[] unicodeBytes = unicode.GetBytes(unicodeString);
            byte[] asciiBytes = Encoding.Convert(unicode, ascii, unicodeBytes);

            char[] asciiChars = new char[ascii.GetCharCount(asciiBytes, 0, asciiBytes.Length)];
            ascii.GetChars(asciiBytes, 0, asciiBytes.Length, asciiChars, 0);
            string asciiString = new string(asciiChars);
            Console.WriteLine("Original string: {0}", unicodeString);
            Console.WriteLine("Ascii converted string: {0}", asciiString);
            //pipeServer.Read(buff, 0, 1024);
            */
            string[] coord;
            string[] circles = File.ReadAllLines(@"S:/Unity/Projects/FYDP-Unity/colour_coordinates.txt");
            char[] split = new char[2];
            split[0] = '-';
            split[1] = ',';
            foreach (string circle in circles)
            {
                coord = circle.Split(split);
                float ball_X = Convert.ToSingle(coord[1]);
                Console.WriteLine(ball_X);
                Console.ReadLine();
            }
        }
    }
}
