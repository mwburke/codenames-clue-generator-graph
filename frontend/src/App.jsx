import React, { useState, useMemo } from 'react';

export default function App() {
  const [image, setImage] = useState(null), [imageFile, setImageFile] = useState(null);
  const [apiKey, setApiKey] = useState(''), [method, setMethod] = useState('neo4j');
  const [status, setStatus] = useState('idle'), [boardState, setBoardState] = useState(null), [clues, setClues] = useState([]);
  const [selectedTargets, setSelectedTargets] = useState([]);

  // Flatten the boardState into a renderable array of word objects
  const boardWords = useMemo(() => {
    if (!boardState) return [];
    const words = [];
    (boardState.my_team || boardState.myTeam || []).forEach(w => words.push({ word: w, role: 'team' }));
    (boardState.opponent || []).forEach(w => words.push({ word: w, role: 'opponent' }));
    (boardState.neutral || []).forEach(w => words.push({ word: w, role: 'neutral' }));
    if (boardState.assassin) words.push({ word: boardState.assassin, role: 'assassin' });

    // Sort alphabetically so the layout looks intentional
    return words.sort((a, b) => a.word.localeCompare(b.word));
  }, [boardState]);

  const handleProcess = async () => {
    if (!imageFile || !apiKey) return;
    setStatus('parsing');
    try {
      const formData = new FormData(); formData.append('image', imageFile); formData.append('api_key', apiKey);
      const parseRes = await fetch('http://localhost:8000/api/parse-board', { method: 'POST', body: formData });
      const parsedState = await parseRes.json(); setBoardState(parsedState); setStatus('generating clues...');

      const cluesRes = await fetch('http://localhost:8000/api/generate-clues', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ my_team: parsedState.my_team || parsedState.myTeam, opponent: parsedState.opponent, neutral: parsedState.neutral, assassin: parsedState.assassin, method, api_key: apiKey })
      });
      setClues((await cluesRes.json()).clues); setStatus('idle');
    } catch (e) { setStatus('idle'); }
  };

  const loadExampleBoard = async () => {
    try {
      setStatus('loading example...');
      const res = await fetch('http://localhost:8000/api/load-salt-board');
      const parsedState = await res.json();
      setBoardState(parsedState);
      setImage(null);
      setImageFile(null);

      // Automatically generate clues for it without image
      setStatus('generating clues...');
      const cluesRes = await fetch('http://localhost:8000/api/generate-clues', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          my_team: parsedState.my_team || parsedState.myTeam,
          opponent: parsedState.opponent,
          neutral: parsedState.neutral,
          assassin: parsedState.assassin,
          method,
          api_key: apiKey
        })
      });
      if (!cluesRes.ok) {
        const errorData = await cluesRes.json();
        console.error("Backend Error:", errorData.detail);
        setClues([]);
        setStatus('idle');
        return;
      }
      setClues((await cluesRes.json()).clues);
      setStatus('idle');
    } catch (e) {
      setStatus('idle');
      console.error("Failed to load example:", e);
    }
  };

  const handleClueHover = (targets) => {
    setSelectedTargets(targets);
  };

  return (
    <div className="p-4 md:p-8 min-h-screen text-slate-200">
      <div className="max-w-6xl mx-auto space-y-8">

        <header className="flex flex-col md:flex-row justify-between items-center glass-panel p-6 rounded-2xl">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              Codenames AI
            </h1>
            <p className="text-slate-400 text-sm mt-1">Graph & Semantic Clue Generator</p>
          </div>

          <div className="flex gap-4 mt-4 md:mt-0">
            <select value={method} onChange={(e) => setMethod(e.target.value)} className="bg-slate-800 border border-slate-700 text-slate-200 p-2 rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none">
              <option value="graph">Graph Only (Fast)</option>
              <option value="semantic-first">Semantic First (High Quality)</option>
              <option value="hybrid">Hybrid (Fusion)</option>
              <option value="neo4j">Neo4j (Requires DB)</option>
            </select>
            <input type="password" placeholder="Gemini API Key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="bg-slate-800 border border-slate-700 text-slate-200 p-2 rounded-lg focus:ring-2 focus:ring-cyan-500 outline-none" />
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          <div className="lg:col-span-1 space-y-6">
            <div className="glass-panel p-6 rounded-2xl">
              <h2 className="text-xl font-semibold mb-4 text-slate-100 flex items-center gap-2">
                <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                Board Input
              </h2>

              <div className="space-y-4">
                <div className="relative group">
                  <input type="file" className="block w-full text-sm text-slate-400 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:font-semibold file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 transition cursor-pointer" onChange={(e) => { setImage(URL.createObjectURL(e.target.files[0])); setImageFile(e.target.files[0]); }} />
                </div>
                {image && <img src={image} className="rounded-lg w-full max-h-40 object-cover border border-slate-700 shadow-inner" />}

                <div className="flex flex-col gap-3 pt-2">
                  <button onClick={handleProcess} disabled={!imageFile || !apiKey} className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 disabled:from-slate-700 disabled:to-slate-800 disabled:text-slate-500 text-white font-medium py-3 rounded-xl shadow-lg hover:shadow-cyan-500/25 transition-all duration-300">
                    {status === 'idle' ? 'Parse Image & Generate' : status.toUpperCase()}
                  </button>
                  <div className="relative flex items-center py-2">
                    <div className="flex-grow border-t border-slate-700"></div>
                    <span className="flex-shrink-0 mx-4 text-slate-500 text-sm">OR</span>
                    <div className="flex-grow border-t border-slate-700"></div>
                  </div>
                  <button onClick={loadExampleBoard} className="w-full bg-slate-800 border border-slate-700 text-slate-300 font-medium py-3 rounded-xl hover:bg-slate-700 hover:text-white transition-all duration-300">
                    {status.includes('loading') ? 'Loading...' : 'Load Scenario from Dataset'}
                  </button>
                </div>
              </div>
            </div>

            <div className="glass-panel p-6 rounded-2xl flex flex-col max-h-[600px]">
              <h2 className="text-xl font-semibold mb-4 text-slate-100 flex items-center gap-2">
                <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                Generated Clues
              </h2>

              {clues.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-slate-500 italic text-sm py-10">
                  {status.includes('generating') ? 'Thinking...' : 'No clues generated yet.'}
                </div>
              ) : (
                <div className="space-y-3 overflow-y-auto pr-2" style={{ scrollbarWidth: 'thin' }}>
                  {clues.map((c, i) => (
                    <div
                      key={i}
                      onMouseEnter={() => handleClueHover(c.targets)}
                      onMouseLeave={() => handleClueHover([])}
                      className="group bg-slate-800/50 border border-slate-700/50 hover:border-cyan-500/50 p-4 rounded-xl cursor-pointer transition-all duration-200"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex flex-col">
                          <span className="font-bold text-xl text-white group-hover:text-cyan-300 transition-colors uppercase tracking-wider">
                            {c.clue.split(' (from:')[0]} <span className="text-slate-400 text-base font-normal ml-1">({c.number})</span>
                          </span>
                          <span className="text-[10px] text-slate-500 italic mt-0.5 lowercase">
                            source: {c.raw_clue || c.candidate}
                          </span>
                        </div>
                        <span className="bg-slate-900 border border-slate-700 text-cyan-400 text-xs font-mono px-2 py-1 rounded-md">
                          Score: {c.score.toFixed(1)}
                        </span>
                      </div>
                      <div className="text-sm text-slate-400 flex flex-wrap gap-1">
                        {c.targets.map(t => (
                          <span key={t} className="bg-cyan-900/30 text-cyan-200 px-2 py-0.5 rounded text-xs">{t}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 glass-panel p-6 md:p-8 rounded-3xl h-fit">
            <div className="flex justify-between items-end mb-6">
              <h2 className="text-2xl font-semibold text-slate-100">Active Board</h2>
              {boardWords.length > 0 && <span className="text-sm text-slate-400">{boardWords.length} words remaining</span>}
            </div>

            {boardWords.length === 0 ? (
              <div className="h-64 flex items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl">
                <p className="text-slate-500">Parse an image or load an example to view the board</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 md:gap-4">
                {boardWords.map((item, idx) => {
                  const isTargeted = selectedTargets.includes(item.word);
                  const roleClasses = {
                    team: 'bg-cyan-900/50 text-cyan-200 border-cyan-500/40 hover:bg-cyan-800/70 shadow-[0_0_15px_rgba(6,182,212,0.15)]',
                    opponent: 'bg-rose-900/50 text-rose-200 border-rose-500/40 hover:bg-rose-800/70 shadow-[0_0_15px_rgba(225,29,72,0.15)]',
                    neutral: 'bg-slate-800/50 text-slate-300 border-slate-600/40 hover:bg-slate-700/70',
                    assassin: 'bg-black text-slate-400 border-slate-800 hover:text-white hover:border-red-500 hover:shadow-[0_0_20px_rgba(239,68,68,0.3)]'
                  };
                  return (
                    <div
                      key={idx}
                      className={`word-card border ${roleClasses[item.role]} ${isTargeted ? 'card-targeted' : ''}`}
                    >
                      {item.word.toUpperCase()}
                      <div className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-current opacity-50"></div>
                    </div>
                  );
                })}
              </div>
            )}

            {boardWords.length > 0 && (
              <div className="flex flex-wrap gap-4 mt-8 pt-6 border-t border-slate-800 text-sm font-medium">
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-cyan-500 shadow-[0_0_8px_#06b6d4]"></div> Team</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-rose-500 shadow-[0_0_8px_#e11d48]"></div> Opponent</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-slate-500"></div> Neutral</div>
                <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-black border border-slate-700"></div> Assassin</div>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
